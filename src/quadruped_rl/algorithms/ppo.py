"""PPO — reference implementation (clipped surrogate + optional adaptive KL).

Supports both env kinds:
  - single BaseEnv (numpy I/O)      -> RolloutBuffer path
  - VectorEnv (torch I/O, N envs)   -> VecRolloutBuffer path (Isaac Lab)

Thesis spec (1.1.2): clipped objective with adaptive KL penalty in parallel;
GAE-lambda adjusted by terrain complexity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from quadruped_rl.algorithms.base import Algorithm, RolloutBuffer, VecRolloutBuffer
from quadruped_rl.algorithms.networks import Critic, GaussianActor
from quadruped_rl.envs.base_env import VectorEnv
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
        self.gae_lambda = gae_lambda_for_terrain(
            a["gae_lambda"], cfg.get("terrain", {}).get("category", "baseline")
        )
        self.buffer: RolloutBuffer | VecRolloutBuffer | None = None
        self.kl_penalty = float(a.get("adaptive_kl", {}).get("penalty_init", 0.0))

    # ------------------------------------------------------------------ utils
    def _to_t(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs, deterministic: bool = False):
        """Single obs [obs_dim] -> np action; batched [N, obs_dim] -> tensor."""
        obs_t = self._to_t(obs)
        batched = obs_t.ndim == 2
        if not batched:
            obs_t = obs_t.unsqueeze(0)
        dist = self.actor.dist(obs_t)
        action = dist.mean if deterministic else dist.sample()
        if batched:
            return action
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------- collection paths
    def collect_and_update(self, env, obs):
        if isinstance(env, VectorEnv):
            return self._collect_vec(env, obs)
        return self._collect_single(env, obs)

    def _collect_single(self, env, obs):
        a = self.acfg
        if self.buffer is None:
            self.buffer = RolloutBuffer(
                a["rollout_steps"], self.obs_dim, self.act_dim, a["gamma"], self.gae_lambda
            )
        steps = 0
        while not self.buffer.full:
            with torch.no_grad():
                obs_t = self._to_t(obs).unsqueeze(0)
                dist = self.actor.dist(obs_t)
                action_t = dist.sample()
                log_prob = float(dist.log_prob(action_t).sum(-1))
                value = float(self.critic(obs_t))
            action = action_t.squeeze(0).cpu().numpy()
            next_obs, reward, done, _ = env.step(action)
            self.buffer.add(obs, action, reward, done, value, log_prob)
            obs = env.reset() if done else next_obs
            steps += 1
        with torch.no_grad():
            last_value = float(self.critic(self._to_t(obs).unsqueeze(0)))
        adv, ret = self.buffer.compute_returns(last_value)
        metrics = self._update(
            self._to_t(self.buffer.obs),
            self._to_t(self.buffer.actions),
            self._to_t(self.buffer.log_probs),
            self._to_t(adv),
            self._to_t(ret),
        )
        return obs, metrics, steps

    def _collect_vec(self, env: VectorEnv, obs):
        a = self.acfg
        if self.buffer is None:
            self.buffer = VecRolloutBuffer(
                a["rollout_steps"],
                env.num_envs,
                self.obs_dim,
                self.act_dim,
                a["gamma"],
                self.gae_lambda,
                device=self.device,
            )
        obs = self._to_t(obs)
        reward_sum, vel_sum, n_steps = 0.0, 0.0, 0
        while not self.buffer.full:
            with torch.no_grad():
                dist = self.actor.dist(obs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(-1)
                values = self.critic(obs)
            next_obs, rewards, dones, info = env.step(actions)
            next_obs = self._to_t(next_obs)
            rewards = self._to_t(rewards)
            # timeout bootstrapping (legged_gym): a truncated episode is not
            # a true terminal — fold the value estimate back into the reward
            if "time_outs" in info:
                rewards = (
                    rewards + self.acfg["gamma"] * values * self._to_t(info["time_outs"]).float()
                )
            self.buffer.add(obs, actions, rewards, self._to_t(dones).float(), values, log_probs)
            obs = next_obs
            reward_sum += float(rewards.mean())
            if "positions" in info:
                vel_sum += (
                    float(self._to_t(info["forward_velocity"]).mean())
                    if "forward_velocity" in info
                    else 0.0
                )
            n_steps += 1
        with torch.no_grad():
            last_values = self.critic(obs)
        adv, ret = self.buffer.compute_returns(last_values)
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731  [T,N,...] -> [T*N,...]
        metrics = self._update(
            flat(self.buffer.obs),
            flat(self.buffer.actions),
            self.buffer.log_probs.reshape(-1),
            adv.reshape(-1),
            ret.reshape(-1),
        )
        steps = a["rollout_steps"] * env.num_envs
        metrics["mean_reward"] = reward_sum / max(n_steps, 1)
        metrics["actor_std"] = float(self.actor.log_std.exp().mean())
        if vel_sum:
            metrics["mean_forward_velocity"] = vel_sum / max(n_steps, 1)
        return obs, metrics, steps

    # ------------------------------------------------------------------ update
    def _update(self, obs, actions, old_log_probs, adv, returns) -> dict[str, float]:
        a = self.acfg
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

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
                    ratio * adv[mb],
                    ratio.clamp(1 - clip, 1 + clip) * adv[mb],
                ).mean()
                approx_kl = (old_log_probs[mb] - log_probs).mean()
                policy_loss = -surrogate + self.kl_penalty * approx_kl

                value_loss = (self.critic(obs[mb]) - returns[mb]).pow(2).mean()
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

    # ------------------------------------------------------------------- io
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
