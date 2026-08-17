"""Difficulty curriculum: easy -> medium -> hard promotion based on a rolling
success-rate window (configs/default.yaml: curriculum section)."""

from __future__ import annotations

from collections import deque
from typing import Any


class Curriculum:
    def __init__(self, cfg: dict[str, Any]):
        c = cfg["curriculum"]
        self.enabled: bool = c["enabled"]
        self.levels: list[str] = list(c["levels"])
        self.threshold: float = c["promote_success_rate"]
        self.window: deque[bool] = deque(maxlen=c["window_episodes"])
        self._idx = 0

    @property
    def level(self) -> str:
        return self.levels[self._idx]

    def report_episode(self, success: bool) -> bool:
        """Record an episode outcome; returns True if difficulty was promoted."""
        if not self.enabled:
            return False
        self.window.append(success)
        if (
            len(self.window) == self.window.maxlen
            and sum(self.window) / len(self.window) >= self.threshold
            and self._idx < len(self.levels) - 1
        ):
            self._idx += 1
            self.window.clear()
            return True
        return False


def gae_lambda_for_terrain(base_lambda: float, terrain_category: str) -> float:
    """PPO GAE-lambda scheduling by terrain complexity (see thesis 1.1.2):
    harder/noisier terrain -> lower lambda (less variance, more bias)."""
    adjust = {
        "baseline": 0.0,
        "structured": 0.0,
        "irregular": -0.02,
        "composite": -0.03,
        "extreme": -0.05,
    }
    return max(0.8, base_lambda + adjust.get(terrain_category, 0.0))
