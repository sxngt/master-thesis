"""TRPO — Trust Region Policy Optimization (on-policy).

Thesis spec (1.1.2, configs/algorithm/trpo.yaml):
  - natural policy gradient: solve F x = g with conjugate gradient, where
    F is the Fisher information matrix accessed via Hessian-vector products
    of the mean KL (cg_iters, cg_damping)
  - step size: beta = sqrt(2 * max_kl / (x^T F x)); backtracking line search
    (line_search_steps, line_search_backtrack) accepting only steps that
    improve the surrogate while satisfying KL(old || new) <= max_kl
  - value function: separate network fit by Adam regression (value_lr)

Shares RolloutBuffer/GAE and the Gaussian actor/critic with PPO.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from quadruped_rl.algorithms.base import Algorithm, RolloutBuffer, VecRolloutBuffer
from quadruped_rl.algorithms.networks import Critic, GaussianActor
from quadruped_rl.envs.base_env import VectorEnv
from quadruped_rl.registry import register_algorithm


def _flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in module.parameters()])


def _set_flat_params(module: torch.nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[offset : offset + n].view_as(p))
        offset += n


def _flat_grad(
    output: torch.Tensor,
    params: list[torch.nn.Parameter],
    retain_graph: bool = False,
    create_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(
        output, params, create_graph=create_graph, retain_graph=retain_graph or create_graph
    )
    return torch.cat([g.contiguous().view(-1) for g in grads])


def conjugate_gradient(matvec, b: torch.Tensor, iters: int, tol: float = 1e-10) -> torch.Tensor:
    """Solve A x = b for SPD A given only the matrix-vector product."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rs_old = r.dot(r)
    for _ in range(iters):
        Ap = matvec(p)
        alpha = rs_old / (p.dot(Ap) + 1e-12)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r.dot(r)
        if rs_new < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


@register_algorithm("trpo")
class TRPO(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        net = a["network"]
        self.actor = GaussianActor(obs_dim, act_dim, net["actor"]).to(self.device)
        self.critic = Critic(obs_dim, net["critic"]).to(self.device)
        self.value_opt = torch.optim.Adam(self.critic.parameters(), lr=a["value_lr"])
        self.buffer: RolloutBuffer | VecRolloutBuffer | None = None

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

    # ------------------------------------------------------------ collection
    def collect_and_update(self, env, obs):
        if isinstance(env, VectorEnv):
            return self._collect_vec(env, obs)
        return self._collect_single(env, obs)

    def _collect_single(self, env, obs):
        a = self.acfg
        if self.buffer is None:
            self.buffer = RolloutBuffer(
                a["rollout_steps"], self.obs_dim, self.act_dim, a["gamma"], a["gae_lambda"]
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
        adv, returns = self.buffer.compute_returns(last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        metrics = self._update(
            self._to_t(self.buffer.obs),
            self._to_t(self.buffer.actions),
            self._to_t(self.buffer.log_probs),
            self._to_t(adv),
            self._to_t(returns),
        )
        return obs, metrics, steps

    def _collect_vec(self, env, obs):
        a = self.acfg
        if self.buffer is None:
            self.buffer = VecRolloutBuffer(
                a["rollout_steps"],
                env.num_envs,
                self.obs_dim,
                self.act_dim,
                a["gamma"],
                a["gae_lambda"],
                device=self.device,
            )
        obs = self._to_t(obs)
        while not self.buffer.full:
            with torch.no_grad():
                dist = self.actor.dist(obs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(-1)
                values = self.critic(obs)
            next_obs, rewards, dones, _ = env.step(actions)
            next_obs = self._to_t(next_obs)
            self.buffer.add(
                obs, actions, self._to_t(rewards), self._to_t(dones).float(), values, log_probs
            )
            obs = next_obs
        with torch.no_grad():
            last_values = self.critic(obs)
        adv, returns = self.buffer.compute_returns(last_values)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        metrics = self._update(
            flat(self.buffer.obs),
            flat(self.buffer.actions),
            self.buffer.log_probs.reshape(-1),
            adv.reshape(-1),
            returns.reshape(-1),
        )
        steps = a["rollout_steps"] * env.num_envs
        return obs, metrics, steps

    # ---------------------------------------------------------------- update
    def _update(self, obs, actions, old_log_probs, adv_t, ret_t) -> dict[str, float]:
        a = self.acfg
        params = list(self.actor.parameters())

        with torch.no_grad():
            old_dist = self.actor.dist(obs)
            old_mean = old_dist.mean.clone()
            old_std = old_dist.stddev.clone()

        def surrogate() -> torch.Tensor:
            dist = self.actor.dist(obs)
            log_probs = dist.log_prob(actions).sum(-1)
            return ((log_probs - old_log_probs).exp() * adv_t).mean()

        def mean_kl() -> torch.Tensor:
            dist = self.actor.dist(obs)
            fixed = torch.distributions.Normal(old_mean, old_std)
            return torch.distributions.kl_divergence(fixed, dist).sum(-1).mean()

        # policy gradient of the surrogate
        loss = surrogate()
        g = _flat_grad(loss, params, retain_graph=True).detach()

        # Fisher-vector product via double backprop of the KL
        kl = mean_kl()
        kl_grad = _flat_grad(kl, params, create_graph=True)

        def fvp(v: torch.Tensor) -> torch.Tensor:
            hv = _flat_grad(kl_grad.dot(v), params, retain_graph=True).detach()
            return hv + a["cg_damping"] * v

        step_dir = conjugate_gradient(fvp, g, a["cg_iters"])
        shs = step_dir.dot(fvp(step_dir))
        step_size = torch.sqrt(2.0 * a["max_kl"] / (shs + 1e-12))
        full_step = step_size * step_dir

        # backtracking line search
        old_params = _flat_params(self.actor)
        surrogate_before = float(loss.detach())
        accepted, kl_after, surrogate_after = False, 0.0, surrogate_before
        frac = 1.0
        for _ in range(a["line_search_steps"]):
            _set_flat_params(self.actor, old_params + frac * full_step)
            with torch.no_grad():
                surrogate_after = float(surrogate())
                kl_after = float(mean_kl())
            if kl_after <= a["max_kl"] and surrogate_after > surrogate_before:
                accepted = True
                break
            frac *= a["line_search_backtrack"]
        if not accepted:
            _set_flat_params(self.actor, old_params)
            kl_after, surrogate_after = 0.0, surrogate_before

        # value function regression
        value_losses = []
        for _ in range(5):
            value_loss = (self.critic(obs) - ret_t).pow(2).mean()
            self.value_opt.zero_grad()
            value_loss.backward()
            self.value_opt.step()
            value_losses.append(float(value_loss.detach()))

        return {
            "surrogate_improvement": surrogate_after - surrogate_before,
            "kl": kl_after,
            "step_accepted": float(accepted),
            "value_loss": float(np.mean(value_losses)),
        }

    # -------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "value_opt": self.value_opt.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.value_opt.load_state_dict(ckpt["value_opt"])
