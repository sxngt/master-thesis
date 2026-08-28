"""A3C — Asynchronous Advantage Actor-Critic (on-policy).

Thesis spec (1.1.2, configs/algorithm/a3c.yaml): 16-32 asynchronous workers
sharing a global network, n-step advantage updates, gradient accumulation.

Harness adaptation: each worker is a thread with its OWN env instance
(created from the registry with an offset seed; worker 0 reuses the
Trainer's env to preserve its seeding/curriculum). Within one
collect_and_update() call every worker asynchronously performs
`gradient_accumulation` rollouts of `rollout_steps`, computes gradients on
its local model copy, and applies them to the shared global network under a
lock as it finishes (Hogwild-with-lock); workers only join at the call
boundary. Classic n-step returns (no GAE), entropy bonus, grad-norm clip.

Not supported with VectorEnv backends: A3C's async CPU-worker design is
fundamentally mismatched with a single GPU-vectorized simulator — use
PPO/TRPO there. LSTM actor path is pending (docs/roadmap.md).
"""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quadruped_rl.algorithms.base import Algorithm
from quadruped_rl.algorithms.networks import Critic, GaussianActor
from quadruped_rl.envs.base_env import VectorEnv
from quadruped_rl.registry import get_env_backend, register_algorithm


@register_algorithm("a3c")
class A3C(Algorithm):
    def __init__(self, cfg: dict[str, Any], obs_dim: int, act_dim: int):
        super().__init__(cfg, obs_dim, act_dim)
        a = self.acfg
        net = a["network"]
        actor_spec = dict(net["actor"])
        if actor_spec.get("recurrent"):
            actor_spec["recurrent"] = False  # LSTM path pending (roadmap)
        self._actor_spec, self._critic_spec = actor_spec, net["critic"]
        self.actor = GaussianActor(obs_dim, act_dim, actor_spec).to(self.device)
        self.critic = Critic(obs_dim, self._critic_spec).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=a["learning_rate"]
        )
        self._lock = threading.Lock()
        self._workers: list[dict[str, Any]] | None = None

    def _to_t(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        dist = self.actor.dist(self._to_t(obs).unsqueeze(0))
        action = dist.mean if deterministic else dist.sample()
        return action.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------- workers
    def _init_workers(self, env) -> None:
        num_workers = self.acfg["num_workers"]
        workers = [{"env": env, "obs": env.reset()}]  # worker 0: Trainer's env
        backend = get_env_backend(self.cfg["sim"]["backend"])
        for w in range(1, num_workers):
            w_cfg = copy.deepcopy(self.cfg)
            w_cfg["run"]["seed"] = self.cfg["run"]["seed"] + 1000 * w
            w_env = backend(w_cfg)
            workers.append({"env": w_env, "obs": w_env.reset()})
        for w in workers:
            w["actor"] = GaussianActor(self.obs_dim, self.act_dim, self._actor_spec).to(self.device)
            w["critic"] = Critic(self.obs_dim, self._critic_spec).to(self.device)
        self._workers = workers

    def _worker_round(self, w: dict[str, Any], metrics_out: list) -> None:
        a = self.acfg
        for _ in range(a["gradient_accumulation"]):
            with self._lock:  # sync local <- global
                w["actor"].load_state_dict(self.actor.state_dict())
                w["critic"].load_state_dict(self.critic.state_dict())

            obs_l, act_l, rew_l, done_l = [], [], [], []
            obs = w["obs"]
            for _ in range(a["rollout_steps"]):
                with torch.no_grad():
                    dist = w["actor"].dist(self._to_t(obs).unsqueeze(0))
                    action = dist.sample().squeeze(0).cpu().numpy()
                next_obs, reward, done, _ = w["env"].step(action)
                obs_l.append(obs)
                act_l.append(action)
                rew_l.append(reward)
                done_l.append(float(done))
                obs = w["env"].reset() if done else next_obs
            w["obs"] = obs

            # classic n-step returns with bootstrap
            with torch.no_grad():
                bootstrap = float(w["critic"](self._to_t(obs).unsqueeze(0)))
            returns = np.zeros(len(rew_l), dtype=np.float32)
            running = bootstrap
            for t in reversed(range(len(rew_l))):
                running = rew_l[t] + a["gamma"] * running * (1.0 - done_l[t])
                returns[t] = running

            obs_t = self._to_t(np.asarray(obs_l))
            act_t = self._to_t(np.asarray(act_l))
            ret_t = self._to_t(returns)
            values = w["critic"](obs_t)
            adv = (ret_t - values).detach()

            dist = w["actor"].dist(obs_t)
            log_probs = dist.log_prob(act_t).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            policy_loss = -(log_probs * adv).mean()
            value_loss = (values - ret_t).pow(2).mean()
            loss = policy_loss + a["value_coef"] * value_loss - a["entropy_coef"] * entropy

            w["actor"].zero_grad()
            w["critic"].zero_grad()
            loss.backward()
            local_params = list(w["actor"].parameters()) + list(w["critic"].parameters())
            torch.nn.utils.clip_grad_norm_(local_params, a["max_grad_norm"])

            with self._lock:  # async apply: local grads -> global params
                global_params = list(self.actor.parameters()) + list(self.critic.parameters())
                for gp, lp in zip(global_params, local_params, strict=True):
                    gp.grad = lp.grad.clone()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            metrics_out.append(float(loss.detach()))

    # ------------------------------------------------------------ collection
    def collect_and_update(self, env, obs):
        if isinstance(env, VectorEnv):
            raise NotImplementedError(
                "A3C's asynchronous CPU-worker design does not map to "
                "GPU-vectorized backends; use PPO/TRPO with Isaac Lab."
            )
        a = self.acfg
        if self._workers is None:
            self._init_workers(env)
        self._workers[0]["obs"] = obs  # stay consistent with the Trainer loop

        losses: list[float] = []
        threads = [
            threading.Thread(target=self._worker_round, args=(w, losses)) for w in self._workers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        steps = a["num_workers"] * a["gradient_accumulation"] * a["rollout_steps"]
        metrics = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "num_workers": float(a["num_workers"]),
        }
        return self._workers[0]["obs"], metrics, steps

    # -------------------------------------------------------------------- io
    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
