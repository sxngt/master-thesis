"""PyBullet backend — cross-validation simulator #1 (CPU, high portability).

Used to verify that policies trained in Isaac Gym transfer across physics
engines (sim-to-sim gap, thesis 3.3). Physics params are aligned to measured
values via system identification (configs/sim/pybullet.yaml).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pybullet  # noqa: F401

from quadruped_rl.envs.base_env import BaseEnv
from quadruped_rl.registry import register_env_backend


@register_env_backend("pybullet")
class PyBulletEnv(BaseEnv):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        raise NotImplementedError(
            "PyBulletEnv: implement following the BaseEnv contract "
            "(see backends/mock.py); load robot URDF via pybullet.loadURDF, "
            "terrain via createCollisionShape(GEOM_HEIGHTFIELD)."
        )

    @property
    def observation_dim(self) -> int:
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
