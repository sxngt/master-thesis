"""Deterministic mock environment — no simulator required.

Purpose: smoke tests, CI, and harness/algorithm plumbing verification.
Never used for reported results. A trivial point-mass 'robot' whose forward
progress depends on the mean action, with noise, so learning signal exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quadruped_rl.envs.base_env import BaseEnv, VectorEnv
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


@register_env_backend("mock_vec")
class MockVectorEnv(VectorEnv):
    """Vectorized mock backend (CPU torch tensors) — exercises the exact
    VectorEnv contract Isaac Lab uses, without a simulator. CI/tests only."""

    OBS_DIM = 48
    ACT_DIM = 12

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        import torch

        self.torch = torch
        self._device = "cpu"
        self.gen = torch.Generator().manual_seed(cfg["run"]["seed"])
        self.goal_distance_m = 5.0
        self._pos = torch.zeros(self.num_envs, 3)
        self._t = torch.zeros(self.num_envs, dtype=torch.long)
        # tunable reward params (coach hooks); drive_gain scales progress so a
        # scheduler's change is observable in the KPIs
        self._params = {"drive.weight": 1.0, "action_cost.weight": 0.01}
        self._stat_steps = 0
        self._stat_reward = 0.0

    def reward_params(self) -> dict[str, float]:
        return dict(self._params)

    def set_reward_params(self, updates: dict[str, float]) -> None:
        for k, v in updates.items():
            if k not in self._params:
                raise KeyError(k)
            self._params[k] = float(v)

    def training_stats(self) -> dict[str, float]:
        if self._stat_steps == 0:
            return {}
        out = {
            "reward/total": self._stat_reward / self._stat_steps,
            "gait/duty_factor": 0.6,
            "gait/diagonal_sync": 0.9,
        }
        self._stat_steps, self._stat_reward = 0, 0.0
        return out

    @property
    def observation_dim(self) -> int:
        return self.OBS_DIM

    @property
    def action_dim(self) -> int:
        return self.ACT_DIM

    @property
    def device(self) -> str:
        return self._device

    def _obs(self):
        return self.torch.randn(self.num_envs, self.OBS_DIM, generator=self.gen)

    def reset(self):
        self._pos.zero_()
        self._t.zero_()
        return self._obs()

    def step(self, actions):
        torch = self.torch
        actions = actions.detach().cpu()
        self._t += 1
        drive = torch.tanh(actions.mean(dim=-1)) * self._params["drive.weight"]
        self._pos[:, 0] += drive.clamp(min=0.0) * self.control_dt
        rewards = drive - self._params["action_cost.weight"] * actions.square().mean(dim=-1)
        self._stat_steps += 1
        self._stat_reward += float(rewards.mean())
        reached = self._pos[:, 0] >= self.goal_distance_m
        dones = (self._t >= self.max_steps) | reached
        n = self.num_envs
        info = {
            "positions": self._pos.clone(),
            "orientations_rpy": torch.randn(n, 3, generator=self.gen) * 0.02,
            "torques": actions.clone(),
            "joint_velocities": torch.randn(n, self.ACT_DIM, generator=self.gen) * 0.1,
            "contact_forces": torch.randn(n, 4, generator=self.gen).abs() * 30.0,
            "power_w": actions.abs().sum(dim=-1) * 5.0,
            "falls": torch.zeros(n),
            "reached_goal": reached.clone(),
            "goal_distance_m": torch.full((n,), self.goal_distance_m),
        }
        # auto-reset finished envs
        if dones.any():
            idx = dones.nonzero(as_tuple=True)[0]
            self._pos[idx] = 0.0
            self._t[idx] = 0
        return self._obs(), rewards, dones, info
