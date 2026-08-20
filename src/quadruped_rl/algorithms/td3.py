"""TD3 — Twin Delayed DDPG.

Spec (configs/algorithm/td3.yaml): target policy smoothing (target_noise,
noise_clip), delayed policy updates (policy_delay=2), clipped double Q.
Structure follows sac.py (deterministic actor + 2 Q-critics + targets).

STATUS: skeleton — implement before Phase 1 completion (see docs/roadmap.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quadruped_rl.algorithms.base import Algorithm, ReplayBuffer
from quadruped_rl.registry import register_algorithm


@register_algorithm("td3")
class TD3(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        self.buffer = ReplayBuffer(
            self.acfg["buffer_size"], obs_dim, act_dim, seed=cfg["run"]["seed"]
        )
        raise NotImplementedError("TD3: implement following algorithms/sac.py structure")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def collect_and_update(self, env, obs):
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError
