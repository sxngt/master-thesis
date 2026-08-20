"""DDPG.

Spec (configs/algorithm/ddpg.yaml): Ornstein-Uhlenbeck exploration noise vs.
parameter-space noise — both variants required for the comparison experiment
(noise_type switch). Structure follows sac.py minus entropy machinery.

STATUS: skeleton — implement before Phase 1 completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quadruped_rl.algorithms.base import Algorithm, ReplayBuffer
from quadruped_rl.registry import register_algorithm


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, dim: int, theta: float, sigma: float, seed: int = 0):
        self.dim, self.theta, self.sigma = dim, theta, sigma
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(dim)

    def reset(self) -> None:
        self.state = np.zeros(self.dim)

    def sample(self) -> np.ndarray:
        self.state += -self.theta * self.state + self.sigma * self.rng.standard_normal(self.dim)
        return self.state


@register_algorithm("ddpg")
class DDPG(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        self.noise = OUNoise(act_dim, a["ou_theta"], a["ou_sigma"], seed=cfg["run"]["seed"])
        self.buffer = ReplayBuffer(a["buffer_size"], obs_dim, act_dim, seed=cfg["run"]["seed"])
        raise NotImplementedError("DDPG: implement following algorithms/sac.py structure")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def collect_and_update(self, env, obs):
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError
