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
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from quadruped_rl.llm_feedback.prompts import (
    TRAJECTORY_EVAL_SYSTEM,
    TRAJECTORY_EVAL_USER,
)
from quadruped_rl.llm_feedback.schemas import LLMRewardOutput

log = logging.getLogger(__name__)


def _load_dotenv() -> None:
    """Load the repo-root .env (API keys) once; silently no-op if absent."""
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[3] / ".env")
    except Exception:  # pragma: no cover - optional convenience
        pass


class LLMClient:
    """Thin provider-agnostic chat wrapper (anthropic | openai).

    `reasoning_effort` is forwarded to OpenAI reasoning models (gpt-5.x / o-series);
    for those the sampling temperature is fixed by the API, so determinism is
    approximated by prompt caching + full prompt/response logging instead.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-sonnet-5",
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ):
        _load_dotenv()
        self.provider = provider
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.json_mode = json_mode
        self.last_usage: dict[str, int] = {}
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
            self.last_usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
            return resp.content[0].text
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        u = resp.usage
        self.last_usage = {
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "reasoning_tokens": getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0,
        }
        return resp.choices[0].message.content or ""


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
