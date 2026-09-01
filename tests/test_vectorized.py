"""Vectorized path: VecRolloutBuffer GAE, reward parity, mock_vec contract,
vectorized PPO end-to-end — all CPU, no simulator."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quadruped_rl.algorithms.base import RolloutBuffer, VecRolloutBuffer  # noqa: E402
from quadruped_rl.harness.config import compose_config  # noqa: E402
from quadruped_rl.rewards.traditional import TraditionalReward  # noqa: E402
from quadruped_rl.rewards.vectorized import VectorizedTraditionalReward  # noqa: E402


def test_vec_gae_matches_single_env_gae():
    """VecRolloutBuffer with N=1 must reproduce RolloutBuffer's GAE exactly."""
    rng = np.random.default_rng(0)
    h = 16
    single = RolloutBuffer(h, 4, 2, gamma=0.99, gae_lambda=0.95)
    vec = VecRolloutBuffer(h, 1, 4, 2, gamma=0.99, gae_lambda=0.95)
    for _ in range(h):
        obs, act = rng.normal(size=4), rng.normal(size=2)
        r, d, v, lp = rng.normal(), float(rng.random() < 0.1), rng.normal(), rng.normal()
        single.add(obs, act, r, d, v, lp)
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, -1)  # noqa: E731
        vec.add(
            t(obs),
            t(act),
            torch.tensor([r], dtype=torch.float32),
            torch.tensor([d], dtype=torch.float32),
            torch.tensor([v], dtype=torch.float32),
            torch.tensor([lp], dtype=torch.float32),
        )
    last = rng.normal()
    adv_s, ret_s = single.compute_returns(float(last))
    adv_v, ret_v = vec.compute_returns(torch.tensor([last], dtype=torch.float32))
    np.testing.assert_allclose(adv_v.squeeze(-1).numpy(), adv_s, rtol=1e-5)
    np.testing.assert_allclose(ret_v.squeeze(-1).numpy(), ret_s, rtol=1e-5)


def test_vectorized_reward_matches_traditional():
    """Definition parity: torch-vectorized components == numpy reference."""
    cfg = compose_config(reward="traditional")
    ref = TraditionalReward(cfg["reward"])
    vec = VectorizedTraditionalReward(cfg["reward"])
    rng = np.random.default_rng(1)
    n = 32
    batch = {
        "forward_velocity_ms": rng.normal(1.0, 0.5, n),
        "torques": rng.normal(0, 5, (n, 12)),
        "orientation_rpy": rng.normal(0, 0.2, (n, 3)),
        "foot_slip_velocity": rng.normal(0, 0.3, (n, 4)),
        "joint_limit_violation": rng.normal(-0.1, 0.1, (n, 12)),
        "action_delta": rng.normal(0, 0.1, (n, 12)),
        "fallen": rng.random(n) < 0.2,
        "feet_first_contact": (rng.random((n, 4)) < 0.3).astype(np.float64),
        "feet_last_air_time": rng.uniform(0.0, 1.0, (n, 4)),
        "command_speed": np.full(n, 1.0),
        "lateral_velocity_ms": rng.normal(0, 0.3, n),
        "yaw_rate_rads": rng.normal(0, 0.5, n),
    }
    t_batch = {
        k: torch.as_tensor(v if k == "fallen" else np.asarray(v, np.float32))
        for k, v in batch.items()
    }
    total_vec, _ = vec(t_batch)
    for i in range(n):
        state = {k: (bool(v[i]) if k == "fallen" else v[i]) for k, v in batch.items()}
        state["command_speed"] = float(state["command_speed"])
        state["lateral_velocity_ms"] = float(state["lateral_velocity_ms"])
        state["yaw_rate_rads"] = float(state["yaw_rate_rads"])
        total_ref, _ = ref(state)
        assert np.isclose(float(total_vec[i]), total_ref, rtol=1e-4), f"env {i}"


def _vec_cfg(num_envs=8, **run_overrides):
    return compose_config(
        algorithm="ppo",
        robot="a1",
        terrain="flat",
        reward="traditional",
        overrides={
            "sim": {"backend": "mock_vec", "num_envs": num_envs},
            "run": {"seed": 0, "device": "cpu", **run_overrides},
            "logging": {"wandb": False},
        },
    )


def test_mock_vec_contract():
    from quadruped_rl.registry import get_env_backend

    env = get_env_backend("mock_vec")(_vec_cfg())
    obs = env.reset()
    assert obs.shape == (8, env.observation_dim)
    actions = torch.zeros(8, env.action_dim)
    obs, rewards, dones, info = env.step(actions)
    assert rewards.shape == (8,) and dones.shape == (8,)
    for key in (
        "positions",
        "orientations_rpy",
        "torques",
        "joint_velocities",
        "contact_forces",
        "power_w",
        "falls",
        "reached_goal",
        "goal_distance_m",
    ):
        assert key in info, key
        assert len(info[key]) == 8


def test_vectorized_ppo_training(tmp_path):
    """Full Trainer loop on the vectorized mock env (VecRolloutBuffer path)."""
    from quadruped_rl.harness.trainer import Trainer

    cfg = _vec_cfg(
        num_envs=8,
        total_timesteps=2_000,
        eval_interval_steps=1_000,
        checkpoint_interval_steps=1_000,
        eval_episodes=4,
    )
    result = Trainer(cfg, run_dir=tmp_path / "run").train()
    assert "success_rate" in result["final"]
    assert (tmp_path / "run" / "config.yaml").exists()


def test_isaaclab_backend_not_registered_without_isaaclab():
    """On machines without isaaclab the backend must be silently absent."""
    import importlib.util

    from quadruped_rl.registry import _ENV_BACKENDS

    if importlib.util.find_spec("isaaclab") is None:
        assert "isaaclab" not in _ENV_BACKENDS
    else:
        assert "isaaclab" in _ENV_BACKENDS
