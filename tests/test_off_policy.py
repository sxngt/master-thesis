"""Off-policy algorithms (SAC/TD3/DDPG): mechanics + end-to-end smoke."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quadruped_rl.harness.config import compose_config  # noqa: E402
from quadruped_rl.harness.trainer import Trainer  # noqa: E402
from quadruped_rl.registry import get_algorithm  # noqa: E402

FAST_OFFPOLICY = {
    "warmup_steps": 64,
    "steps_per_iteration": 128,
    "batch_size": 32,
    "buffer_size": 10_000,
    "network": {
        "actor": {"hidden": [32, 32], "activation": "relu"},
        "critic": {"hidden": [32, 32], "activation": "relu"},
    },
}


def _cfg(algorithm, algo_overrides=None):
    return compose_config(
        algorithm=algorithm,
        robot="a1",
        terrain="flat",
        reward="traditional",
        overrides={
            "algorithm": {**FAST_OFFPOLICY, **(algo_overrides or {})},
            "sim": {"backend": "mock", "num_envs": 1, "episode_length_s": 2.0},
            "run": {
                "seed": 0,
                "device": "cpu",
                "total_timesteps": 600,
                "eval_interval_steps": 100_000,
                "checkpoint_interval_steps": 100_000,
                "eval_episodes": 2,
            },
            "logging": {"wandb": False},
        },
    )


@pytest.mark.parametrize("algorithm", ["sac", "td3", "ddpg"])
def test_off_policy_smoke_training(algorithm, tmp_path):
    result = Trainer(_cfg(algorithm), run_dir=tmp_path / algorithm).train()
    assert "success_rate" in result["final"]
    assert (tmp_path / algorithm / "checkpoints" / "best.pt").exists()


def test_ddpg_parameter_space_noise_smoke(tmp_path):
    cfg = _cfg("ddpg", {"noise_type": "parameter_space"})
    result = Trainer(cfg, run_dir=tmp_path / "ddpg_ps").train()
    assert "success_rate" in result["final"]


def test_td3_actions_bounded_and_deterministic_repeatable():
    cfg = _cfg("td3")
    algo = get_algorithm("td3")(cfg, obs_dim=48, act_dim=12)
    obs = np.random.default_rng(0).standard_normal(48).astype(np.float32)
    a1 = algo.act(obs, deterministic=True)
    a2 = algo.act(obs, deterministic=True)
    np.testing.assert_array_equal(a1, a2)
    assert np.all(np.abs(algo.act(obs)) <= 1.0)  # exploration stays clipped


def test_td3_policy_delay():
    """Actor/target update only every `policy_delay` critic updates."""
    cfg = _cfg("td3", {"policy_delay": 2})
    algo = get_algorithm("td3")(cfg, obs_dim=8, act_dim=2)
    rng = np.random.default_rng(0)
    for _ in range(64):
        algo.buffer.add(
            rng.standard_normal(8),
            rng.standard_normal(2),
            rng.standard_normal(),
            rng.standard_normal(8),
            0.0,
        )
    before = [p.clone() for p in algo.actor.parameters()]
    algo._update()  # call 1: critic only
    assert all(torch.equal(b, p) for b, p in zip(before, algo.actor.parameters(), strict=True))
    algo._update()  # call 2: actor updates
    assert any(not torch.equal(b, p) for b, p in zip(before, algo.actor.parameters(), strict=True))


def test_ddpg_ou_noise_temporal_correlation():
    from quadruped_rl.algorithms.ddpg import OUNoise

    noise = OUNoise(4, theta=0.15, sigma=0.2, seed=0)
    samples = np.stack([noise.sample() for _ in range(500)])
    # OU is mean-reverting around 0 with bounded variance
    assert abs(samples.mean()) < 0.2
    # consecutive samples are correlated (unlike white noise)
    corr = np.corrcoef(samples[:-1, 0], samples[1:, 0])[0, 1]
    assert corr > 0.5
    noise.reset()
    assert np.all(noise.state == 0.0)


def test_ddpg_param_noise_perturbs_and_adapts():
    cfg = _cfg("ddpg", {"noise_type": "parameter_space", "param_noise_stddev": 0.1})
    algo = get_algorithm("ddpg")(cfg, obs_dim=8, act_dim=2)
    obs = np.zeros(8, dtype=np.float32)
    clean = algo.act(obs, deterministic=True)
    noisy = algo.act(obs, deterministic=False)
    assert not np.array_equal(clean, noisy)  # perturbed copy differs
    rng = np.random.default_rng(0)
    for _ in range(64):
        algo.buffer.add(
            rng.standard_normal(8), rng.standard_normal(2), 0.0, rng.standard_normal(8), 0.0
        )
    sigma_before = algo.param_sigma
    algo._adapt_param_sigma()
    assert algo.param_sigma != sigma_before  # adaptation moved sigma
