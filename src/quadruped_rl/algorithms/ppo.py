"""PPO — reference implementation (clipped surrogate + optional adaptive KL).

Serves as the template for the Algorithm interface; other algorithms follow
this file's structure. Thesis spec (1.1.2): clipped objective with adaptive
KL penalty in parallel; GAE-lambda adjusted by terrain complexity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from quadruped_rl.algorithms.base import Algorithm, RolloutBuffer
from quadruped_rl.algorithms.networks import Critic, GaussianActor
from quadruped_rl.envs.curriculum import gae_lambda_for_terrain
from quadruped_rl.registry import register_algorithm


@register_algorithm("ppo")
class PPO(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        net = a["network"]
        self.actor = GaussianActor(obs_dim, act_dim, net["actor"]).to(self.device)
        self.critic = Critic(obs_dim, net["critic"]).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=a["learning_rate"],
        )
        gae_lambda = gae_lambda_for_terrain(
            a["gae_lambda"], cfg.get("terrain", {}).get("category", "baseline")
        )
        self.buffer = RolloutBuffer(a["rollout_steps"], obs_dim, act_dim, a["gamma"], gae_lambda)
        self.kl_penalty = float(a.get("adaptive_kl", {}).get("penalty_init", 0.0))

    def _to_t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        dist = self.actor.dist(self._to_t(obs).unsqueeze(0))
        action = dist.mean if deterministic else dist.sample()
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def _policy_step(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        obs_t = self._to_t(obs).unsqueeze(0)
        dist = self.actor.dist(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs_t)
        return (action.squeeze(0).cpu().numpy(), float(log_prob), float(value))

    def collect_and_update(self, env, obs):
        steps = 0
        while not self.buffer.full:
            action, log_prob, value = self._policy_step(obs)
            next_obs, reward, done, _ = env.step(action)
            self.buffer.add(obs, action, reward, done, value, log_prob)
            obs = env.reset() if done else next_obs
            steps += 1
        with torch.no_grad():
            last_value = float(self.critic(self._to_t(obs).unsqueeze(0)))
        metrics = self._update(last_value)
        return obs, metrics, steps

    def _update(self, last_value: float) -> dict[str, float]:
        a = self.acfg
        adv, returns = self.buffer.compute_returns(last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = self._to_t(self.buffer.obs)
        actions = self._to_t(self.buffer.actions)
        old_log_probs = self._to_t(self.buffer.log_probs)
        adv_t, ret_t = self._to_t(adv), self._to_t(returns)

        n = len(obs)
        mb_size = max(n // a["num_minibatches"], 1)
        idx = np.arange(n)
        losses, kls = [], []

        for _ in range(a["num_epochs"]):
            np.random.shuffle(idx)
            for start in range(0, n, mb_size):
                mb = idx[start : start + mb_size]
                dist = self.actor.dist(obs[mb])
                log_probs = dist.log_prob(actions[mb]).sum(-1)
                ratio = (log_probs - old_log_probs[mb]).exp()

                clip = a["clip_range"]
                surrogate = torch.min(
                    ratio * adv_t[mb],
                    ratio.clamp(1 - clip, 1 + clip) * adv_t[mb],
                ).mean()
                approx_kl = (old_log_probs[mb] - log_probs).mean()
                policy_loss = -surrogate + self.kl_penalty * approx_kl

                value_loss = (self.critic(obs[mb]) - ret_t[mb]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = policy_loss + a["value_coef"] * value_loss - a["entropy_coef"] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    a["max_grad_norm"],
                )
                self.optimizer.step()
                losses.append(float(loss.detach()))
                kls.append(float(approx_kl.detach()))

        mean_kl = float(np.mean(kls))
        akl = a.get("adaptive_kl", {})
        if akl.get("enabled"):
            target = akl["target_kl"]
            if mean_kl > 1.5 * target:
                self.kl_penalty *= 2.0
            elif mean_kl < target / 1.5:
                self.kl_penalty *= 0.5
        return {"loss": float(np.mean(losses)), "approx_kl": mean_kl, "kl_penalty": self.kl_penalty}

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "kl_penalty": self.kl_penalty,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.kl_penalty = ckpt.get("kl_penalty", self.kl_penalty)
