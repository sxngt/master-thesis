"""Hybrid reward: R_total = alpha*R_traditional + beta*R_LLM + gamma*R_pref.

- R_traditional: numeric components (traditional.py), per-step.
- R_LLM: LLM-generated language-grounded reward, evaluated per trajectory
  segment and distributed over steps (llm_feedback/translator.py).
- R_pref: learned human-preference reward model (llm_feedback/reward_model.py).
- alpha/beta/gamma: initialized from config, optimized by meta-learning.
"""

from __future__ import annotations

from typing import Any

from quadruped_rl.rewards.traditional import TraditionalReward


class HybridReward:
    def __init__(self, reward_cfg: dict[str, Any], llm_scorer=None, preference_model=None):
        self.alpha = float(reward_cfg["alpha"])
        self.beta = float(reward_cfg["beta"])
        self.gamma = float(reward_cfg["gamma"])
        trad_cfg = reward_cfg["traditional"]
        if "components" not in trad_cfg and "reward" in trad_cfg:
            trad_cfg = trad_cfg["reward"]  # {inherit: reward/traditional.yaml} wrapping
        self.traditional = TraditionalReward(trad_cfg)
        self.llm_scorer = llm_scorer  # llm_feedback.translator.LLMRewardScorer
        self.preference_model = preference_model

    def step_reward(self, state: dict[str, Any]) -> tuple[float, dict[str, float]]:
        r_trad, breakdown = self.traditional(state)
        r_pref = 0.0
        if self.preference_model is not None:
            r_pref = float(self.preference_model.score_state(state))
        total = self.alpha * r_trad + self.gamma * r_pref
        breakdown.update(r_traditional=r_trad, r_preference=r_pref)
        return total, breakdown

    def segment_bonus(self, trajectory_summary: dict[str, Any]) -> float:
        """LLM reward on a trajectory segment (sparse; added at segment end)."""
        if self.llm_scorer is None:
            return 0.0
        return self.beta * float(self.llm_scorer.score(trajectory_summary))

    def set_weights(self, alpha: float, beta: float, gamma: float) -> None:
        """Called by the meta-learning weight optimizer."""
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
