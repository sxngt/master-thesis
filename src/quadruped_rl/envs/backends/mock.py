"""Deterministic mock environment — no simulator required.

Purpose: smoke tests, CI, and harness/algorithm plumbing verification.
Never used for reported results. A trivial point-mass 'robot' whose forward
progress depends on the mean action, with noise, so learning signal exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quadruped_rl.envs.base_env import BaseEnv
from quadruped_rl.registry import register_env_backend


@register_env_backend("mock")
class MockEnv(BaseEnv):
    OBS_DIM = 48
    ACT_DIM = 12

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg["run"]["seed"])
        self.goal_distance_m = 5.0
        self._t = 0
        self._pos = np.zeros(3)

    @property
    def observation_dim(self) -> int:
        return self.OBS_DIM

    @property
    def action_dim(self) -> int:
        return self.ACT_DIM

    def reset(self) -> np.ndarray:
        self._t = 0
        self._pos = np.zeros(3)
        return self.rng.standard_normal(self.OBS_DIM).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self._t += 1
        drive = float(np.tanh(np.mean(action)))
        self._pos = self._pos + np.array([max(drive, 0.0) * self.control_dt, 0.0, 0.0])
        obs = self.rng.standard_normal(self.OBS_DIM).astype(np.float32)
        reward = drive - 0.01 * float(np.mean(np.square(action)))
        reached = self._pos[0] >= self.goal_distance_m
        done = self._t >= self.max_steps or reached
        info = {
            "positions": self._pos.copy(),
            "orientations_rpy": self.rng.standard_normal(3) * 0.02,
            "torques": action.astype(np.float64),
            "joint_velocities": self.rng.standard_normal(self.ACT_DIM) * 0.1,
            "contact_forces": np.abs(self.rng.standard_normal(4)) * 30.0,
            "power_w": float(np.sum(np.abs(action))) * 5.0,
            "falls": 0,
            "reached_goal": bool(reached),
            "goal_distance_m": self.goal_distance_m,
        }
        return obs, reward, done, info
