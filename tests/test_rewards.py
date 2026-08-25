"""Reward composition and component registry."""

import numpy as np

from quadruped_rl.harness.config import compose_config
from quadruped_rl.rewards.traditional import TraditionalReward


def _state(fallen=False, v=1.0):
    return {
        "forward_velocity_ms": v,
        "torques": np.zeros(12),
        "orientation_rpy": np.zeros(3),
        "foot_slip_velocity": np.zeros(4),
        "joint_limit_violation": np.zeros(12),
        "action_delta": np.zeros(12),
        "fallen": fallen,
    }


def test_traditional_reward_at_target_velocity():
    cfg = compose_config(reward="traditional")
    reward = TraditionalReward(cfg["reward"])
    total, breakdown = reward(_state(v=1.0))
    # perfect tracking + alive bonus, all penalties zero
    assert np.isclose(breakdown["forward_velocity"], 1.0)
    assert np.isclose(breakdown["alive_bonus"], 0.1)
    assert np.isclose(total, 1.1)


def test_fall_penalty_dominates():
    cfg = compose_config(reward="traditional")
    reward = TraditionalReward(cfg["reward"])
    total_ok, _ = reward(_state(fallen=False))
    total_fall, breakdown = reward(_state(fallen=True))
    assert breakdown["termination"] == -10.0
    assert total_fall < total_ok


def test_hybrid_weights_from_config():
    from quadruped_rl.rewards.hybrid import HybridReward

    cfg = compose_config(reward="hybrid_llm")
    hybrid = HybridReward(cfg["reward"])
    total, breakdown = hybrid.step_reward(_state())
    assert "r_traditional" in breakdown
    assert hybrid.segment_bonus({}) == 0.0  # no LLM scorer attached
