"""VectorEnv collection paths for SAC/TD3/DDPG/TRPO (roadmap #5) — the
routes used for Isaac Lab training. All CPU via the mock_vec backend."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quadruped_rl.algorithms.base import ReplayBuffer  # noqa: E402
from quadruped_rl.harness.config import compose_config  # noqa: E402
from quadruped_rl.harness.trainer import Trainer  # noqa: E402
from quadruped_rl.registry import get_algorithm, get_env_backend  # noqa: E402

SMALL = {
    "warmup_steps": 64,
    "steps_per_iteration": 128,
    "batch_size": 32,
    "buffer_size": 10_000,
    "network": {
        "actor": {"hidden": [32, 32], "activation": "relu"},
        "critic": {"hidden": [32, 32], "activation": "relu"},
    },
}


def _cfg(algorithm, algo_overrides=None, num_envs=8):
    return compose_config(
        algorithm=algorithm,
        robot="a1",
        terrain="flat",
        reward="traditional",
        overrides={
            "algorithm": {**SMALL, **(algo_overrides or {})},
            "sim": {"backend": "mock_vec", "num_envs": num_envs, "episode_length_s": 2.0},
            "run": {
                "seed": 0,
                "device": "cpu",
                "total_timesteps": 1_500,
                "eval_interval_steps": 100_000,
                "checkpoint_interval_steps": 100_000,
                "eval_episodes": 4,
            },
            "logging": {"wandb": False},
        },
    )


def test_replay_add_batch_and_wraparound():
    buf = ReplayBuffer(capacity=10, obs_dim=3, act_dim=2, seed=0)
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((8, 3)).astype(np.float32)
    buf.add_batch(
        obs,
        rng.standard_normal((8, 2)),
        rng.standard_normal(8),
        rng.standard_normal((8, 3)),
        np.zeros(8),
    )
    assert buf.size == 8 and buf.ptr == 8
    obs2 = rng.standard_normal((6, 3)).astype(np.float32)
    buf.add_batch(
        obs2,
        rng.standard_normal((6, 2)),
        rng.standard_normal(6),
        rng.standard_normal((6, 3)),
        np.ones(6),
    )
    assert buf.size == 10 and buf.ptr == 4  # wrapped: 8+6 = 14 % 10
    np.testing.assert_array_equal(buf.obs[0], obs2[2])  # wrapped rows land at 0..3
    assert buf.sample(5)["obs"].shape == (5, 3)


def test_replay_add_batch_accepts_torch_tensors():
    buf = ReplayBuffer(capacity=100, obs_dim=3, act_dim=2, seed=0)
    buf.add_batch(
        torch.randn(4, 3), torch.randn(4, 2), torch.randn(4), torch.randn(4, 3), torch.zeros(4)
    )
    assert buf.size == 4


@pytest.mark.parametrize(
    "algorithm,overrides",
    [
        ("sac", None),
        ("td3", None),
        ("ddpg", None),  # OU noise
        ("ddpg", {"noise_type": "parameter_space"}),
        ("trpo", {"rollout_steps": 16}),
    ],
)
def test_vec_smoke_training(algorithm, overrides, tmp_path):
    cfg = _cfg(algorithm, overrides)
    result = Trainer(cfg, run_dir=tmp_path / "run").train()
    assert "success_rate" in result["final"]
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()


@pytest.mark.parametrize("algorithm", ["sac", "td3", "ddpg", "trpo", "ppo"])
def test_batched_act_shapes(algorithm):
    cfg = _cfg(algorithm)
    algo = get_algorithm(algorithm)(cfg, obs_dim=48, act_dim=12)
    batch = torch.randn(8, 48)
    out = algo.act(batch, deterministic=True)
    assert isinstance(out, torch.Tensor) and out.shape == (8, 12)
    single = algo.act(np.zeros(48, dtype=np.float32), deterministic=True)
    assert isinstance(single, np.ndarray) and single.shape == (12,)


def test_vec_transition_count_matches_report():
    """Reported step count must equal transitions actually inserted."""
    cfg = _cfg("sac", num_envs=8)
    env = get_env_backend("mock_vec")(cfg)
    algo = get_algorithm("sac")(cfg, env.observation_dim, env.action_dim)
    obs = env.reset()
    _, _, steps = algo.collect_and_update(env, obs)
    assert steps == algo.buffer.size == algo._total_steps
    assert steps == max(1, 128 // 8) * 8


def test_ddpg_vec_ou_resets_on_done():
    cfg = _cfg("ddpg")
    algo = get_algorithm("ddpg")(cfg, obs_dim=48, act_dim=12)
    algo._vec_ou_sample(4, "cpu")
    assert algo._vec_ou_state.abs().sum() > 0
    dones = torch.tensor([True, False, True, False])
    algo._vec_ou_reset(dones)
    assert torch.all(algo._vec_ou_state[0] == 0)
    assert torch.all(algo._vec_ou_state[2] == 0)
    assert algo._vec_ou_state[1].abs().sum() > 0
