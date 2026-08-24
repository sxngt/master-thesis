"""LLM API gateway: the ONLY module that calls LLM providers (see CLAUDE.md).

Converts natural-language feedback + trajectory summaries into validated
numeric reward signals. Includes caching (identical segment summaries are
scored once) and hard-fail isolation (malformed responses are discarded).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

from pydantic import ValidationError

from quadruped_rl.llm_feedback.prompts import (
    TRAJECTORY_EVAL_SYSTEM,
    TRAJECTORY_EVAL_USER,
)
from quadruped_rl.llm_feedback.schemas import LLMRewardOutput

log = logging.getLogger(__name__)


class LLMClient:
    """Thin provider-agnostic chat wrapper (anthropic | openai)."""

    def __init__(self, provider: str = "anthropic", model: str = "claude-sonnet-5"):
        self.provider = provider
        self.model = model
        if provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        elif provider == "openai":
            import openai

            self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        else:
            raise ValueError(f"Unknown provider '{provider}'")

    def complete(self, system: str, user: str, max_tokens: int = 1024) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content


def _extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in LLM response")
    return json.loads(text[start : end + 1])


class LLMRewardScorer:
    """Scores trajectory segments; used by rewards.hybrid.HybridReward."""

    def __init__(self, llm_cfg: dict[str, Any], client: LLMClient | None = None):
        self.cfg = llm_cfg
        self.client = client or LLMClient(llm_cfg["provider"], llm_cfg["model"])
        self.cache: dict[str, float] = {} if llm_cfg.get("cache", True) else None
        self.feedback_snippets: list[str] = []
        self.discarded = 0

    def add_feedback_context(self, snippets: list[str]) -> None:
        self.feedback_snippets = snippets[-20:]  # keep the context bounded

    def score(self, trajectory_summary: dict[str, Any]) -> float:
        key = hashlib.sha256(
            json.dumps(trajectory_summary, sort_keys=True, default=str).encode()
        ).hexdigest()
        if self.cache is not None and key in self.cache:
            return self.cache[key]

        user = TRAJECTORY_EVAL_USER.format(
            terrain_description=trajectory_summary.get("terrain", "unknown"),
            metrics_json=json.dumps(trajectory_summary.get("metrics", {}), indent=2),
            feedback_snippets="\n".join(f"- {s}" for s in self.feedback_snippets) or "(none)",
        )
        try:
            raw = self.client.complete(TRAJECTORY_EVAL_SYSTEM, user)
            parsed = LLMRewardOutput.model_validate(_extract_json(raw))
            value = parsed.overall_score
        except (ValidationError, ValueError, json.JSONDecodeError, KeyError) as e:
            # Reward-poisoning guard: discard, log, contribute nothing.
            self.discarded += 1
            log.warning("Discarded malformed LLM reward response: %s", e)
            value = 0.0

        if self.cache is not None:
            self.cache[key] = value
        return value
