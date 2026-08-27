"""SAC — off-policy reference implementation.

Thesis spec (1.1.2): automatic entropy temperature tuning, twin Q-networks
against overestimation. Serves as the template for TD3/DDPG structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quadruped_rl.algorithms.base import Algorithm, ReplayBuffer
from quadruped_rl.algorithms.networks import QCritic, mlp
from quadruped_rl.registry import register_algorithm


class SquashedGaussianActor(torch.nn.Module):
    """Tanh-squashed Gaussian policy (state-dependent std)."""

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, obs_dim: int, act_dim: int, spec: dict):
        super().__init__()
        self.body = mlp(obs_dim, spec["hidden"], None, activation=spec.get("activation", "relu"))
        last = spec["hidden"][-1]
        self.mu = torch.nn.Linear(last, act_dim)
        self.log_std = torch.nn.Linear(last, act_dim)

    def forward(self, obs, deterministic=False, with_log_prob=True):
        h = self.body(obs)
        mu = self.mu(h)
        std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        dist = torch.distributions.Normal(mu, std)
        pre = mu if deterministic else dist.rsample()
        action = torch.tanh(pre)
        log_prob = None
        if with_log_prob:
            log_prob = dist.log_prob(pre).sum(-1)
            log_prob -= (2 * (np.log(2) - pre - F.softplus(-2 * pre))).sum(-1)
        return action, log_prob


@register_algorithm("sac")
class SAC(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        net = a["network"]
        self.actor = SquashedGaussianActor(obs_dim, act_dim, net["actor"]).to(self.device)
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

        self.auto_temp = a["auto_temperature"]
        self.log_alpha = torch.tensor(
            float(np.log(a["init_temperature"])), requires_grad=True, device=self.device
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = (
            -float(act_dim) if a["target_entropy"] == "auto" else float(a["target_entropy"])
        )

        self.buffer = ReplayBuffer(a["buffer_size"], obs_dim, act_dim, seed=cfg["run"]["seed"])
        self._total_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, _ = self.actor(obs_t, deterministic=deterministic, with_log_prob=False)
        return action.squeeze(0).cpu().numpy()

    def collect_and_update(self, env, obs):
        a = self.acfg
        metrics: dict[str, float] = {}
        for _ in range(a.get("steps_per_iteration", 256)):
            if self._total_steps < a["warmup_steps"]:
                action = np.random.uniform(-1, 1, self.act_dim).astype(np.float32)
            else:
                action = self.act(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.add(obs, action, reward, next_obs, float(done))
            obs = env.reset() if done else next_obs
            self._total_steps += 1
            if self._total_steps >= a["warmup_steps"]:
                for _ in range(a["updates_per_step"]):
                    metrics = self._update()
        return obs, metrics, a.get("steps_per_iteration", 256)

    def _update(self) -> dict[str, float]:
        a = self.acfg
        batch = self.buffer.sample(min(a["batch_size"], self.buffer.size))
        t = lambda x: torch.as_tensor(x, device=self.device)  # noqa: E731
        obs, actions = t(batch["obs"]), t(batch["actions"])
        rewards, next_obs, dones = t(batch["rewards"]), t(batch["next_obs"]), t(batch["dones"])
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_action, next_logp = self.actor(next_obs)
            q_next = torch.min(self.q1_t(next_obs, next_action), self.q2_t(next_obs, next_action))
            target = rewards + a["gamma"] * (1 - dones) * (q_next - alpha * next_logp)

        q_loss = F.mse_loss(self.q1(obs, actions), target) + F.mse_loss(
            self.q2(obs, actions), target
        )
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        new_action, logp = self.actor(obs)
        q_new = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (alpha * logp - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = torch.tensor(0.0)
        if self.auto_temp:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        with torch.no_grad():
            for src, dst in [(self.q1, self.q1_t), (self.q2, self.q2_t)]:
                for p, p_t in zip(src.parameters(), dst.parameters(), strict=True):
                    p_t.mul_(1 - a["tau"]).add_(a["tau"] * p)

        return {
            "q_loss": float(q_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
            "alpha": float(alpha),
            "alpha_loss": float(alpha_loss.detach()),
        }

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_t.load_state_dict(ckpt["q1"])
        self.q2_t.load_state_dict(ckpt["q2"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
