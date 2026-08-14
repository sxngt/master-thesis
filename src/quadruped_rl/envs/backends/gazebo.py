"""Gazebo backend — cross-validation simulator #2 (ROS bridge).

Highest-fidelity contact/sensor simulation; also the staging environment
for Phase 5 hardware deployment (same ROS interfaces as the real A1).
Connects to a running Gazebo instance via ROS 2 topics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import rclpy  # noqa: F401  — ROS 2 required

from quadruped_rl.envs.base_env import BaseEnv
from quadruped_rl.registry import register_env_backend


@register_env_backend("gazebo")
class GazeboEnv(BaseEnv):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        raise NotImplementedError(
            "GazeboEnv: implement ROS 2 pub/sub bridge for joint commands "
            "and sensor states following the BaseEnv contract."
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
