"""TRPO (natural gradient mechanics) and A3C (async workers)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quadruped_rl.algorithms.trpo import (  # noqa: E402
    _flat_params,
    _set_flat_params,
    conjugate_gradient,
)
from quadruped_rl.harness.config import compose_config  # noqa: E402
from quadruped_rl.harness.trainer import Trainer  # noqa: E402
from quadruped_rl.registry import get_algorithm  # noqa: E402

SMALL_NET = {
    "network": {
        "actor": {"hidden": [32, 32], "activation": "tanh"},
        "critic": {"hidden": [32, 32], "activation": "tanh"},
    }
}


def _cfg(algorithm, algo_overrides=None, **run_overrides):
    return compose_config(
        algorithm=algorithm,
        robot="a1",
        terrain="flat",
        reward="traditional",
        overrides={
            "algorithm": {**SMALL_NET, **(algo_overrides or {})},
            "sim": {"backend": "mock", "num_envs": 1, "episode_length_s": 2.0},
            "run": {
                "seed": 0,
                "device": "cpu",
                "total_timesteps": 400,
                "eval_interval_steps": 100_000,
                "checkpoint_interval_steps": 100_000,
                "eval_episodes": 2,
                **run_overrides,
            },
            "logging": {"wandb": False},
        },
    )


# ------------------------------------------------------------------- TRPO
def test_conjugate_gradient_solves_spd_system():
    rng = np.random.default_rng(0)
    m = rng.standard_normal((8, 8))
    a_mat = torch.as_tensor(m @ m.T + 8 * np.eye(8), dtype=torch.float32)
    b = torch.as_tensor(rng.standard_normal(8), dtype=torch.float32)
    x = conjugate_gradient(lambda v: a_mat @ v, b, iters=50)
    expected = torch.linalg.solve(a_mat, b)
    torch.testing.assert_close(x, expected, rtol=1e-3, atol=1e-4)


def test_flat_params_roundtrip():
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2))
    flat = _flat_params(net)
    _set_flat_params(net, flat * 2.0)
    torch.testing.assert_close(_flat_params(net), flat * 2.0)


def test_trpo_update_respects_kl_constraint():
    cfg = _cfg("trpo", {"rollout_steps": 64, "max_kl": 0.01})
    algo = get_algorithm("trpo")(cfg, obs_dim=8, act_dim=2)
    rng = np.random.default_rng(0)
    for _ in range(64):
        algo.buffer.add(
            rng.standard_normal(8),
            rng.standard_normal(2),
            rng.standard_normal(),
            0.0,
            rng.standard_normal(),
            rng.standard_normal() * 0.1,
        )
    metrics = algo._update(last_value=0.0)
    if metrics["step_accepted"]:
        assert metrics["kl"] <= 0.01 + 1e-6
        assert metrics["surrogate_improvement"] > 0.0
    else:  # rejected step must leave the policy untouched
        assert metrics["surrogate_improvement"] == 0.0


def test_trpo_rejected_step_restores_params():
    """Zero advantages -> the surrogate cannot improve -> every line-search
    candidate is rejected and the policy must be restored exactly."""
    cfg = _cfg("trpo", {"rollout_steps": 32, "line_search_steps": 3})
    algo = get_algorithm("trpo")(cfg, obs_dim=8, act_dim=2)
    rng = np.random.default_rng(1)
    for _ in range(32):
        algo.buffer.add(
            rng.standard_normal(8), rng.standard_normal(2), 0.0, 0.0, 0.0, 0.0
        )  # reward=0, value=0 -> adv=0
    before = _flat_params(algo.actor).clone()
    metrics = algo._update(last_value=0.0)
    assert metrics["step_accepted"] == 0.0
    assert metrics["surrogate_improvement"] == 0.0
    torch.testing.assert_close(_flat_params(algo.actor), before)


def test_trpo_smoke_training(tmp_path):
    cfg = _cfg("trpo", {"rollout_steps": 64})
    result = Trainer(cfg, run_dir=tmp_path / "trpo").train()
    assert "success_rate" in result["final"]
    assert (tmp_path / "trpo" / "checkpoints" / "best.pt").exists()


# -------------------------------------------------------------------- A3C
A3C_FAST = {"num_workers": 3, "gradient_accumulation": 2, "rollout_steps": 16}


def test_a3c_smoke_training(tmp_path):
    cfg = _cfg("a3c", A3C_FAST, total_timesteps=300)
    result = Trainer(cfg, run_dir=tmp_path / "a3c").train()
    assert "success_rate" in result["final"]


def test_a3c_workers_update_global_network():
    cfg = _cfg("a3c", A3C_FAST)
    algo = get_algorithm("a3c")(cfg, obs_dim=48, act_dim=12)
    from quadruped_rl.registry import get_env_backend

    env = get_env_backend("mock")(cfg)
    before = [p.clone() for p in algo.actor.parameters()]
    obs, metrics, steps = algo.collect_and_update(env, env.reset())
    assert steps == 3 * 2 * 16
    assert metrics["num_workers"] == 3
    assert len(algo._workers) == 3
    assert any(not torch.equal(b, p) for b, p in zip(before, algo.actor.parameters(), strict=True))


def test_a3c_rejects_vector_env():
    cfg = compose_config(
        algorithm="a3c",
        robot="a1",
        terrain="flat",
        reward="traditional",
        overrides={
            "algorithm": {**SMALL_NET, **A3C_FAST},
            "sim": {"backend": "mock_vec", "num_envs": 4},
            "run": {"seed": 0, "device": "cpu"},
            "logging": {"wandb": False},
        },
    )
    from quadruped_rl.registry import get_env_backend

    env = get_env_backend("mock_vec")(cfg)
    algo = get_algorithm("a3c")(cfg, env.observation_dim, env.action_dim)
    with pytest.raises(NotImplementedError, match="vectorized"):
        algo.collect_and_update(env, env.reset())
