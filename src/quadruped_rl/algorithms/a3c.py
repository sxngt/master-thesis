"""A3C — Asynchronous Advantage Actor-Critic.

Spec (configs/algorithm/a3c.yaml): 16-32 async workers sharing a global
network, gradient accumulation, LSTM actor. Note: within Isaac Gym's
vectorized setting A3C is realized as async worker groups over env shards.

STATUS: skeleton — implement before Phase 1 completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quadruped_rl.algorithms.base import Algorithm
from quadruped_rl.registry import register_algorithm


@register_algorithm("a3c")
class A3C(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        raise NotImplementedError("A3C: implement async worker groups + shared network")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def collect_and_update(self, env, obs):
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError
