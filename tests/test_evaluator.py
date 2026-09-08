"""Evaluator: vectorized rollout must sample a fixed set of envs' first
episodes, not the first episodes to *finish* (race bias)."""

import numpy as np
import torch

from quadruped_rl.envs.base_env import VectorEnv
from quadruped_rl.harness.evaluator import Evaluator


class _StaggeredEnv(VectorEnv):
    """Odd envs fall at step 2 (failure); even envs reach the goal at step 6.

    A "first K to finish" harvest with K <= N/2 sees only the falls
    (success 0.0); the true per-env success rate is 0.5.
    """

    OBS = 4

    def __init__(self, cfg):
        super().__init__(cfg)
        self._t = torch.zeros(self.num_envs, dtype=torch.long)
        self.steps_called = 0

    @property
    def observation_dim(self):
        return self.OBS

    @property
    def action_dim(self):
        return 2

    @property
    def device(self):
        return "cpu"

    def reset(self):
        self._t.zero_()
        return torch.zeros(self.num_envs, self.OBS)

    def step(self, actions):
        n = self.num_envs
        self._t += 1
        self.steps_called += 1
        odd = torch.arange(n) % 2 == 1
        fall = odd & (self._t == 2)
        reached = ~odd & (self._t == 6)
        dones = fall | reached | (self._t >= self.max_steps)
        pos = torch.zeros(n, 3)
        pos[:, 0] = self._t.float() * 0.1
        info = {
            "positions": pos,
            "orientations_rpy": torch.zeros(n, 3),
            "torques": torch.ones(n, 2),
            "joint_velocities": torch.ones(n, 2),
            "contact_forces": torch.ones(n, 4),
            "power_w": torch.ones(n),
            "falls": fall.float(),
            "reached_goal": reached,
            "goal_distance_m": torch.full((n,), 0.6),
        }
        self._t[dones] = 0
        return torch.zeros(n, self.OBS), torch.zeros(n), dones, info


class _Zero:
    def act(self, obs, deterministic=True):
        return torch.zeros(obs.shape[0], 2)


def _cfg(num_envs, eval_episodes):
    return {
        "run": {"eval_episodes": eval_episodes},
        "sim": {"num_envs": num_envs, "dt": 0.01, "control_decimation": 1, "episode_length_s": 0.2},
        "robot": {"mass_kg": 12.0},
    }


def test_vec_rollout_is_not_race_biased():
    cfg = _cfg(num_envs=64, eval_episodes=16)
    env = _StaggeredEnv(cfg)
    out = Evaluator(cfg, env).run(_Zero())
    assert out["success_rate"] == 0.5
    assert out["completion_time_s"] == 6 * env.control_dt


def test_vec_rollout_stops_once_sampled_envs_finished():
    cfg = _cfg(num_envs=64, eval_episodes=16)
    env = _StaggeredEnv(cfg)
    episodes = Evaluator(cfg, env)._rollout_vec(_Zero())
    assert len(episodes) == 16
    assert env.steps_called == 6  # last sampled env finishes at step 6, not max_steps
    assert sorted(e["steps"] for e in episodes) == [2] * 8 + [6] * 8
    # each episode's arrays are sliced to its own length
    assert all(len(e["positions"]) == e["steps"] for e in episodes)


def test_vec_rollout_caps_episode_count_at_num_envs():
    cfg = _cfg(num_envs=8, eval_episodes=100)
    env = _StaggeredEnv(cfg)
    episodes = Evaluator(cfg, env)._rollout_vec(_Zero())
    assert len(episodes) == 8
    assert np.mean([e["reached_goal"] for e in episodes]) == 0.5
