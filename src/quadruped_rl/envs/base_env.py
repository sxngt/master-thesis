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


class VectorEnv(ABC):
    """Vectorized environment contract (GPU-parallel backends: Isaac Lab).

    All tensors are torch (kept on `device` — never forced to CPU in the
    training path). Auto-reset semantics: step() internally resets envs whose
    episode ended and returns the post-reset observation for them, with
    dones[i] = True on the terminal step.

    step() info dict contract (batched; consumed by harness/evaluator.py):
        positions [N, 3], orientations_rpy [N, 3], torques [N, J],
        joint_velocities [N, J], contact_forces [N, num_feet], power_w [N],
        falls [N] (0/1 this step), reached_goal [N], goal_distance_m [N]
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.num_envs: int = cfg["sim"]["num_envs"]
        self.control_dt: float = cfg["sim"]["dt"] * cfg["sim"]["control_decimation"]
        self.max_steps: int = int(cfg["sim"]["episode_length_s"] / self.control_dt)

    @property
    @abstractmethod
    def observation_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> str: ...

    @abstractmethod
    def reset(self):
        """Full reset. Returns obs tensor [N, obs_dim]."""

    @abstractmethod
    def step(self, actions):
        """actions [N, act_dim] -> (obs [N, obs_dim], rewards [N], dones [N], info)."""
