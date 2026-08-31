"""TD3 — Twin Delayed DDPG (off-policy; structure mirrors sac.py).

Thesis spec (1.1.2, configs/algorithm/td3.yaml):
  - target policy smoothing: noise ~ clip(N(0, target_noise), +-noise_clip)
    added to the target action before Q-target evaluation
  - delayed policy updates: actor (and targets) updated every
    `policy_delay` critic updates (2:1)
  - clipped double Q-learning: target = r + gamma * min(Q1', Q2')
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quadruped_rl.algorithms.base import Algorithm, ReplayBuffer
from quadruped_rl.algorithms.networks import DeterministicActor, QCritic
from quadruped_rl.envs.base_env import VectorEnv
from quadruped_rl.registry import register_algorithm


@register_algorithm("td3")
class TD3(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        net = a["network"]
        self.actor = DeterministicActor(obs_dim, act_dim, net["actor"]).to(self.device)
        self.actor_t = DeterministicActor(obs_dim, act_dim, net["actor"]).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1 = QCritic(obs_dim, act_dim, net["critic"]).to(self.device)
        self.q2 = QCritic(obs_dim, act_dim, net["critic"]).to(self.device)
        self.q1_t = QCritic(obs_dim, act_dim, net["critic"]).to(self.device)
        self.q2_t = QCritic(obs_dim, act_dim, net["critic"]).to(self.device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())

        lr = a["learning_rate"]
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        self.buffer = ReplayBuffer(a["buffer_size"], obs_dim, act_dim, seed=cfg["run"]["seed"])
        self.rng = np.random.default_rng(cfg["run"]["seed"])
        self._total_steps = 0
        self._update_calls = 0

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """Single obs [obs_dim] -> np action; batched [N, obs_dim] -> tensor."""
        obs_t = (
            obs if isinstance(obs, torch.Tensor) else torch.as_tensor(obs, dtype=torch.float32)
        ).to(self.device)
        if obs_t.ndim == 2:
            action = self.actor(obs_t)
            if not deterministic:
                action = action + torch.randn_like(action) * self.acfg["exploration_noise"]
            return action.clamp(-1.0, 1.0)
        action = self.actor(obs_t.unsqueeze(0)).squeeze(0).cpu().numpy()
        if not deterministic:
            action = action + self.rng.normal(0.0, self.acfg["exploration_noise"], self.act_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def collect_and_update(self, env, obs):
        if isinstance(env, VectorEnv):
            return self._collect_vec(env, obs)
        a = self.acfg
        metrics: dict[str, float] = {}
        for _ in range(a.get("steps_per_iteration", 256)):
            if self._total_steps < a["warmup_steps"]:
                action = self.rng.uniform(-1, 1, self.act_dim).astype(np.float32)
            else:
                action = self.act(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.add(obs, action, reward, next_obs, float(done))
            obs = env.reset() if done else next_obs
            self._total_steps += 1
            if self._total_steps >= a["warmup_steps"]:
                for _ in range(a.get("updates_per_step", 1)):
                    metrics = self._update()
        return obs, metrics, a.get("steps_per_iteration", 256)

    def _collect_vec(self, env, obs):
        """VectorEnv collection (see sac.py note on num_envs sizing)."""
        a = self.acfg
        n = env.num_envs
        metrics: dict[str, float] = {}
        vec_steps = max(1, a.get("steps_per_iteration", 256) // n)
        obs = (
            obs
            if isinstance(obs, torch.Tensor)
            else torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        )
        for _ in range(vec_steps):
            if self._total_steps < a["warmup_steps"]:
                actions = torch.rand(n, self.act_dim, device=obs.device) * 2 - 1
            else:
                with torch.no_grad():
                    actions = self.act(obs, deterministic=False)
            next_obs, rewards, dones, _ = env.step(actions)
            self.buffer.add_batch(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            self._total_steps += n
            if self._total_steps >= a["warmup_steps"]:
                for _ in range(a.get("updates_per_step", 1)):
                    metrics = self._update()
        return obs, metrics, vec_steps * n

    def _update(self) -> dict[str, float]:
        a = self.acfg
        self._update_calls += 1
        batch = self.buffer.sample(min(a["batch_size"], self.buffer.size))
        t = lambda x: torch.as_tensor(x, device=self.device)  # noqa: E731
        obs, actions = t(batch["obs"]), t(batch["actions"])
        rewards, next_obs, dones = t(batch["rewards"]), t(batch["next_obs"]), t(batch["dones"])

        with torch.no_grad():
            # target policy smoothing
            noise = (torch.randn_like(actions) * a["target_noise"]).clamp(
                -a["noise_clip"], a["noise_clip"]
            )
            next_action = (self.actor_t(next_obs) + noise).clamp(-1.0, 1.0)
            q1_next = self.q1_t(next_obs, next_action)
            q2_next = self.q2_t(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) if a["clipped_double_q"] else q1_next
            target = rewards + a["gamma"] * (1 - dones) * q_next

        q_loss = F.mse_loss(self.q1(obs, actions), target) + F.mse_loss(
            self.q2(obs, actions), target
        )
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        actor_loss = torch.tensor(0.0)
        if self._update_calls % a["policy_delay"] == 0:
            # delayed policy update
            actor_loss = -self.q1(obs, self.actor(obs)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            with torch.no_grad():
                for src, dst in [
                    (self.actor, self.actor_t),
                    (self.q1, self.q1_t),
                    (self.q2, self.q2_t),
                ]:
                    for p, p_t in zip(src.parameters(), dst.parameters(), strict=True):
                        p_t.mul_(1 - a["tau"]).add_(a["tau"] * p)

        return {"q_loss": float(q_loss.detach()), "actor_loss": float(actor_loss.detach())}

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_t": self.actor_t.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_t": self.q1_t.state_dict(),
                "q2_t": self.q2_t.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "q_opt": self.q_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_t.load_state_dict(ckpt["actor_t"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_t.load_state_dict(ckpt["q1_t"])
        self.q2_t.load_state_dict(ckpt["q2_t"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q_opt.load_state_dict(ckpt["q_opt"])
