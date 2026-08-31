"""Algorithm interface + shared buffers.

Every algorithm implements `Algorithm` and registers itself via
@register_algorithm("<name>") so the harness can build it from config alone.

On-policy algorithms (PPO/TRPO/A3C) use RolloutBuffer; off-policy
(SAC/TD3/DDPG) use ReplayBuffer. Keep the distinction — do not unify.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class Algorithm(ABC):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        self.cfg = cfg
        self.acfg = cfg["algorithm"]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device = cfg["run"].get("device", "cpu")

    @abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Single-observation action for evaluation/deployment."""

    @abstractmethod
    def collect_and_update(self, env, obs: np.ndarray) -> tuple[np.ndarray, dict[str, float], int]:
        """Collect experience from env starting at obs, run update(s).
        Returns (next_obs, train_metrics, env_steps_collected)."""

    @abstractmethod
    def save(self, path: str | Path) -> None: ...

    @abstractmethod
    def load(self, path: str | Path) -> None: ...

    @classmethod
    def hyperparameter_space(cls, cfg: dict[str, Any]) -> dict[str, Any]:
        """Optuna search space, read from configs/algorithm/<name>.yaml."""
        return cfg.get("search_space", {})


class RolloutBuffer:
    """Fixed-horizon on-policy storage with GAE(lambda) advantage computation."""

    def __init__(self, horizon: int, obs_dim: int, act_dim: int, gamma: float, gae_lambda: float):
        self.h, self.gamma, self.lam = horizon, gamma, gae_lambda
        self.obs = np.zeros((horizon, obs_dim), dtype=np.float32)
        self.actions = np.zeros((horizon, act_dim), dtype=np.float32)
        self.rewards = np.zeros(horizon, dtype=np.float32)
        self.dones = np.zeros(horizon, dtype=np.float32)
        self.values = np.zeros(horizon, dtype=np.float32)
        self.log_probs = np.zeros(horizon, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob) -> None:
        i = self.ptr
        self.obs[i], self.actions[i] = obs, action
        self.rewards[i], self.dones[i] = reward, done
        self.values[i], self.log_probs[i] = value, log_prob
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.h

    def compute_returns(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        adv = np.zeros(self.h, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.h)):
            next_value = last_value if t == self.h - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_value * nonterminal - self.values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        returns = adv + self.values[: self.h]
        self.ptr = 0
        return adv, returns


class ReplayBuffer:
    """Uniform off-policy replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, batch_size)
        return {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }


class VecRolloutBuffer:
    """On-policy storage for vectorized envs: [horizon, num_envs, ...] with
    per-env GAE(lambda). Torch tensors, kept on the env's device."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        gamma: float,
        gae_lambda: float,
        device: str = "cpu",
    ):
        import torch

        self.h, self.n = horizon, num_envs
        self.gamma, self.lam = gamma, gae_lambda
        self.device = device
        self.obs = torch.zeros(horizon, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(horizon, num_envs, act_dim, device=device)
        self.rewards = torch.zeros(horizon, num_envs, device=device)
        self.dones = torch.zeros(horizon, num_envs, device=device)
        self.values = torch.zeros(horizon, num_envs, device=device)
        self.log_probs = torch.zeros(horizon, num_envs, device=device)
        self.ptr = 0

    def add(self, obs, actions, rewards, dones, values, log_probs) -> None:
        i = self.ptr
        self.obs[i], self.actions[i] = obs, actions
        self.rewards[i], self.dones[i] = rewards, dones
        self.values[i], self.log_probs[i] = values, log_probs
        self.ptr += 1

    @property
    def full(self) -> bool:
        return self.ptr >= self.h

    def compute_returns(self, last_values):
        """last_values: [num_envs] bootstrap. Returns (adv, ret) [h, n]."""
        import torch

        adv = torch.zeros_like(self.rewards)
        gae = torch.zeros(self.n, device=self.device)
        for t in reversed(range(self.h)):
            next_values = last_values if t == self.h - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * nonterminal - self.values[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        returns = adv + self.values
        self.ptr = 0
        return adv, returns


def _replay_add_batch(buffer: ReplayBuffer, obs, actions, rewards, next_obs, dones) -> None:
    """Vectorized insert of N transitions (with ring-buffer wraparound).

    Accepts numpy arrays or torch tensors (detached/moved to CPU here) —
    used by the VectorEnv collection paths of SAC/TD3/DDPG.
    """

    def np_of(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    obs, actions = np_of(obs), np_of(actions)
    rewards, next_obs = np_of(rewards), np_of(next_obs)
    dones = np_of(dones).astype(np.float32)
    n = len(obs)
    idx = (buffer.ptr + np.arange(n)) % buffer.capacity
    buffer.obs[idx] = obs
    buffer.actions[idx] = actions
    buffer.rewards[idx] = rewards
    buffer.next_obs[idx] = next_obs
    buffer.dones[idx] = dones
    buffer.ptr = int((buffer.ptr + n) % buffer.capacity)
    buffer.size = int(min(buffer.size + n, buffer.capacity))


ReplayBuffer.add_batch = _replay_add_batch
