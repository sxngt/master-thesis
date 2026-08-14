"""Environment interface all backends implement (Isaac Gym, PyBullet, mock).

step() info dict contract (consumed by harness/evaluator.py):
    positions [3], orientations_rpy [3], torques [J], joint_velocities [J],
    contact_forces [4], power_w (scalar), falls (0/1),
    reached_goal (bool, terminal), goal_distance_m (float)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEnv(ABC):
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.control_dt: float = cfg["sim"]["dt"] * cfg["sim"]["control_decimation"]
        self.max_steps: int = int(cfg["sim"]["episode_length_s"] / self.control_dt)

    @property
    @abstractmethod
    def observation_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @abstractmethod
    def reset(self) -> np.ndarray: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]: ...
