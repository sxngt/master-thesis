"""DDPG (off-policy; structure mirrors sac.py / td3.py).

Thesis spec (1.1.2, configs/algorithm/ddpg.yaml): the exploration strategy
is an experimental variable —
  - noise_type: "ou": temporally correlated Ornstein-Uhlenbeck action noise
  - noise_type: "parameter_space": Gaussian noise on actor weights with
    adaptive stddev (Plappert et al., 2018): a perturbed actor copy collects
    experience; sigma is adapted so the action-space distance between the
    clean and perturbed policies tracks `param_noise_stddev`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quadruped_rl.algorithms.base import Algorithm, ReplayBuffer
from quadruped_rl.algorithms.networks import DeterministicActor, QCritic
from quadruped_rl.registry import register_algorithm


class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, dim: int, theta: float, sigma: float, seed: int = 0):
        self.dim, self.theta, self.sigma = dim, theta, sigma
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(dim)

    def reset(self) -> None:
        self.state = np.zeros(self.dim)

    def sample(self) -> np.ndarray:
        self.state += -self.theta * self.state + self.sigma * self.rng.standard_normal(self.dim)
        return self.state


@register_algorithm("ddpg")
class DDPG(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        net = a["network"]
        self.actor = DeterministicActor(obs_dim, act_dim, net["actor"]).to(self.device)
        self.actor_t = DeterministicActor(obs_dim, act_dim, net["actor"]).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q = QCritic(obs_dim, act_dim, net["critic"]).to(self.device)
        self.q_t = QCritic(obs_dim, act_dim, net["critic"]).to(self.device)
        self.q_t.load_state_dict(self.q.state_dict())

        lr = a["learning_rate"]
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.buffer = ReplayBuffer(a["buffer_size"], obs_dim, act_dim, seed=cfg["run"]["seed"])
        seed = cfg["run"]["seed"]
        self.noise_type = a["noise_type"]
        if self.noise_type == "ou":
            self.ou = OUNoise(act_dim, a["ou_theta"], a["ou_sigma"], seed=seed)
        elif self.noise_type == "parameter_space":
            self.perturbed_actor = DeterministicActor(obs_dim, act_dim, net["actor"]).to(
                self.device
            )
            self.param_sigma = float(a["param_noise_stddev"])
            self._perturb_actor()
        else:
            raise ValueError(f"Unknown noise_type '{self.noise_type}'")
        self._total_steps = 0

    # -------------------------------------------------- parameter-space noise
    @torch.no_grad()
    def _perturb_actor(self) -> None:
        self.perturbed_actor.load_state_dict(self.actor.state_dict())
        for p in self.perturbed_actor.parameters():
            p.add_(torch.randn_like(p) * self.param_sigma)

    @torch.no_grad()
    def _adapt_param_sigma(self) -> float:
        """Adapt sigma so ||pi(s) - pi_perturbed(s)|| tracks the target."""
        if self.buffer.size == 0:
            return 0.0
        batch = self.buffer.sample(min(64, self.buffer.size))
        obs = torch.as_tensor(batch["obs"], device=self.device)
        dist = float(torch.sqrt(F.mse_loss(self.actor(obs), self.perturbed_actor(obs))))
        target = self.acfg["param_noise_stddev"]
        self.param_sigma *= 1.01 if dist < target else 1.0 / 1.01
        return dist

    # ------------------------------------------------------------------- act
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        use_perturbed = not deterministic and self.noise_type == "parameter_space"
        actor = self.perturbed_actor if use_perturbed else self.actor
        action = actor(obs_t).squeeze(0).cpu().numpy()
        if not deterministic and self.noise_type == "ou":
            action = action + self.ou.sample()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------ collection
    def collect_and_update(self, env, obs):
        a = self.acfg
        metrics: dict[str, float] = {}
        rng = self.buffer.rng
        if self.noise_type == "parameter_space":
            # fresh weight perturbation per collection segment
            self._perturb_actor()
        for _ in range(a.get("steps_per_iteration", 256)):
            if self._total_steps < a["warmup_steps"]:
                action = rng.uniform(-1, 1, self.act_dim).astype(np.float32)
            else:
                action = self.act(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.add(obs, action, reward, next_obs, float(done))
            if done:
                obs = env.reset()
                if self.noise_type == "ou":
                    self.ou.reset()
            else:
                obs = next_obs
            self._total_steps += 1
            if self._total_steps >= a["warmup_steps"]:
                for _ in range(a.get("updates_per_step", 1)):
                    metrics = self._update()
        if self.noise_type == "parameter_space":
            metrics["param_noise_dist"] = self._adapt_param_sigma()
            metrics["param_sigma"] = self.param_sigma
        return obs, metrics, a.get("steps_per_iteration", 256)

    # ---------------------------------------------------------------- update
    def _update(self) -> dict[str, float]:
        a = self.acfg
        batch = self.buffer.sample(min(a["batch_size"], self.buffer.size))
        t = lambda x: torch.as_tensor(x, device=self.device)  # noqa: E731
        obs, actions = t(batch["obs"]), t(batch["actions"])
        rewards, next_obs, dones = t(batch["rewards"]), t(batch["next_obs"]), t(batch["dones"])

        with torch.no_grad():
            q_next = self.q_t(next_obs, self.actor_t(next_obs))
            target = rewards + a["gamma"] * (1 - dones) * q_next

        q_loss = F.mse_loss(self.q(obs, actions), target)
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        actor_loss = -self.q(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for src, dst in [(self.actor, self.actor_t), (self.q, self.q_t)]:
                for p, p_t in zip(src.parameters(), dst.parameters(), strict=True):
                    p_t.mul_(1 - a["tau"]).add_(a["tau"] * p)

        return {"q_loss": float(q_loss.detach()), "actor_loss": float(actor_loss.detach())}

    # -------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_t": self.actor_t.state_dict(),
                "q": self.q.state_dict(),
                "q_t": self.q_t.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "q_opt": self.q_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_t.load_state_dict(ckpt["actor_t"])
        self.q.load_state_dict(ckpt["q"])
        self.q_t.load_state_dict(ckpt["q_t"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q_opt.load_state_dict(ckpt["q_opt"])
        if self.noise_type == "parameter_space":
            self._perturb_actor()
