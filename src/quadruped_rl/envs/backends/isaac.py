"""Isaac Gym backend — PRIMARY TRAINING simulator (GPU-parallel, 4096 envs).

Requires the NVIDIA-distributed isaacgym package (docs/setup.md).
Maps configs (robot URDF, terrain heightfield from envs/terrains.py,
sim physics params from configs/sim/isaacgym.yaml) onto isaacgym.gymapi.
"""

from __future__ import annotations

from typing import Any

import isaacgym  # noqa: F401  — must import before torch (isaacgym constraint)
import numpy as np

from quadruped_rl.envs.base_env import BaseEnv
from quadruped_rl.registry import register_env_backend


@register_env_backend("isaacgym")
class IsaacGymEnv(BaseEnv):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        raise NotImplementedError(
            "IsaacGymEnv: implement gym/sim creation, terrain upload from "
            "envs/terrains.py::make_terrain, actor spawning from robot URDF, "
            "and the BaseEnv step/reset contract (see backends/mock.py)."
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
