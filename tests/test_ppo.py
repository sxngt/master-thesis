"""PPO: adaptive KL penalty schedule stays bounded, non-finite minibatches are skipped."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from quadruped_rl.harness.config import compose_config  # noqa: E402
from quadruped_rl.registry import get_algorithm  # noqa: E402

SMALL_NET = {
    "network": {
        "actor": {"hidden": [32, 32], "activation": "tanh"},
        "critic": {"hidden": [32, 32], "activation": "tanh"},
    }
}


def _ppo(**akl):
    cfg = compose_config(
        algorithm="ppo",
        robot="a1",
        terrain="flat",
        reward="traditional",
        overrides={
            "algorithm": {**SMALL_NET, "adaptive_kl": {"enabled": True, "target_kl": 0.01, **akl}},
            "sim": {"backend": "mock", "num_envs": 1},
            "run": {"seed": 0, "device": "cpu"},
            "logging": {"wandb": False},
        },
    )
    return get_algorithm("ppo")(cfg, obs_dim=8, act_dim=2)


def test_default_bounds_come_from_the_algorithm_yaml():
    algo = _ppo()
    assert algo.kl_penalty == 1.0
    assert algo.kl_penalty_min == 2**-6 and algo.kl_penalty_max == 8.0
    # the floor keeps the penalty within reach: a quarter of low-KL updates
    # leaves beta at the floor, and ten overshoots bring it back to the cap
    for _ in range(100):
        algo.adapt_kl_penalty(0.001)
    assert algo.kl_penalty == 2**-6
    assert [algo.adapt_kl_penalty(0.03) for _ in range(10)][-1] == 8.0


def test_penalty_doubles_and_halves_within_bounds():
    algo = _ppo(penalty_init=1.0, penalty_min=1e-3, penalty_max=8.0)
    # a run parked on a plateau: mean KL above target at every update
    seen = [algo.adapt_kl_penalty(0.05) for _ in range(40)]
    assert seen[:3] == [2.0, 4.0, 8.0] and max(seen) == 8.0 and algo.kl_penalty == 8.0
    # a healthy run: KL under target for a long stretch, then one overshoot
    seen = [algo.adapt_kl_penalty(0.001) for _ in range(40)]
    assert min(seen) == 1e-3 and algo.kl_penalty == 1e-3
    assert algo.adapt_kl_penalty(0.05) == 2e-3
    # inside the dead band nothing moves
    assert algo.adapt_kl_penalty(0.01) == 2e-3


def test_penalty_is_untouched_when_adaptive_kl_is_off():
    algo = _ppo(enabled=False, penalty_init=0.5)
    assert algo.adapt_kl_penalty(1.0) == 0.5 and algo.adapt_kl_penalty(0.0) == 0.5


def test_update_clamps_beta_and_skips_non_finite_minibatches():
    algo = _ppo(penalty_init=1e30, penalty_max=32.0)
    n = 32
    obs = torch.randn(n, 8)
    actions = torch.randn(n, 2)
    with torch.no_grad():
        old_log_probs = algo.actor.dist(obs).log_prob(actions).sum(-1)
    adv = torch.randn(n)
    returns = torch.randn(n)
    before = [p.detach().clone() for p in algo.actor.parameters()]
    # an inf return makes the value loss non-finite for the minibatch that holds it
    returns[0] = float("inf")
    out = algo._update(obs, actions, old_log_probs, adv, returns)
    assert out["kl_penalty"] <= 32.0
    assert out.get("nonfinite_minibatches", 0) >= 1
    assert all(torch.isfinite(p).all() for p in algo.actor.parameters())
    assert np.isfinite(out["loss"])
    # with beta back inside its bounds the next update trains the actor again
    returns[0] = 0.0
    out = algo._update(obs, actions, old_log_probs, adv, returns)
    assert "nonfinite_minibatches" not in out and out["kl_penalty"] <= 32.0
    assert any(not torch.equal(a, b) for a, b in zip(before, algo.actor.parameters(), strict=True))


def test_load_clamps_a_runaway_checkpoint(tmp_path):
    algo = _ppo(penalty_init=1.0, penalty_max=32.0)
    algo.kl_penalty = 1e20
    algo.save(tmp_path / "ckpt.pt")
    fresh = _ppo(penalty_init=1.0, penalty_max=32.0)
    fresh.load(tmp_path / "ckpt.pt")
    assert fresh.kl_penalty == 32.0


def test_epoch_loop_stops_once_a_minibatch_kl_runs_away():
    algo = _ppo(penalty_init=1.0, stop_kl=0.2)
    n = 32
    obs = torch.randn(n, 8)
    actions = torch.randn(n, 2)
    adv = torch.randn(n)
    returns = torch.randn(n)
    # stale log-probs from a policy far away from the current one -> KL >> stop_kl
    old_log_probs = torch.full((n,), 50.0)
    before = [p.detach().clone() for p in algo.actor.parameters()]
    out = algo._update(obs, actions, old_log_probs, adv, returns)
    assert out["kl_early_stop"] == 1.0
    assert all(torch.equal(a, b) for a, b in zip(before, algo.actor.parameters(), strict=True))
    # the runaway KL counts as an overshoot: beta doubles, it must not halve
    assert out["approx_kl"] > 0.2 and algo.kl_penalty == 2.0
    # a healthy update never trips it
    with torch.no_grad():
        fresh = algo.actor.dist(obs).log_prob(actions).sum(-1)
    out = algo._update(obs, actions, fresh, adv, returns)
    assert "kl_early_stop" not in out
