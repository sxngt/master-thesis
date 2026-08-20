"""TRPO.

Spec (configs/algorithm/trpo.yaml): natural policy gradient via conjugate
gradient (cg_iters, cg_damping), KL-constrained step with backtracking line
search. Reuses RolloutBuffer/GAE from base.py and networks from ppo.py.

STATUS: skeleton — implement before Phase 1 completion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from quadruped_rl.algorithms.base import Algorithm, RolloutBuffer
from quadruped_rl.registry import register_algorithm


@register_algorithm("trpo")
class TRPO(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        self.buffer = RolloutBuffer(
            a["rollout_steps"], obs_dim, act_dim, a["gamma"], a["gae_lambda"]
        )
        raise NotImplementedError("TRPO: implement (CG natural gradient + line search)")

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        raise NotImplementedError

    def collect_and_update(self, env, obs):
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        raise NotImplementedError
