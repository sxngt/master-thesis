"""Traditional numeric reward: weighted sum of registered components.

Each component is a pure function (state_dict) -> float, registered by name
so configs/reward/traditional.yaml can enable/weight them declaratively.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quadruped_rl.registry import get_reward_component, register_reward_component


@register_reward_component("forward_velocity")
def forward_velocity(s: dict[str, Any], target_ms: float = 1.0, **_) -> float:
    """Tracking reward: exp penalty on deviation from target forward speed."""
    v = float(s.get("forward_velocity_ms", 0.0))
    return float(np.exp(-4.0 * (v - target_ms) ** 2))


@register_reward_component("energy")
def energy(s: dict[str, Any], **_) -> float:
    tau = np.asarray(s.get("torques", 0.0))
    return float(np.sum(np.square(tau)))


@register_reward_component("stability")
def stability(s: dict[str, Any], **_) -> float:
    rpy = np.asarray(s.get("orientation_rpy", [0.0, 0.0, 0.0]))
    return float(rpy[0] ** 2 + rpy[1] ** 2)


@register_reward_component("foot_slip")
def foot_slip(s: dict[str, Any], **_) -> float:
    """Tangential foot velocity while in contact."""
    slip = np.asarray(s.get("foot_slip_velocity", 0.0))
    return float(np.sum(np.square(slip)))


@register_reward_component("joint_limit")
def joint_limit(s: dict[str, Any], **_) -> float:
    margin = np.asarray(s.get("joint_limit_violation", 0.0))
    return float(np.sum(np.clip(margin, 0.0, None)))


@register_reward_component("action_rate")
def action_rate(s: dict[str, Any], **_) -> float:
    da = np.asarray(s.get("action_delta", 0.0))
    return float(np.sum(np.square(da)))


@register_reward_component("alive_bonus")
def alive_bonus(s: dict[str, Any], **_) -> float:
    return 0.0 if s.get("fallen", False) else 1.0


@register_reward_component("termination")
def termination(s: dict[str, Any], **_) -> float:
    return 1.0 if s.get("fallen", False) else 0.0


class TraditionalReward:
    def __init__(self, reward_cfg: dict[str, Any]):
        self.components: list[tuple[str, float, dict]] = []
        for name, params in reward_cfg["components"].items():
            params = dict(params)
            weight = params.pop("weight")
            # config uses *_ms style kwargs; pass the rest through
            self.components.append((name, weight, params))

    def __call__(self, state: dict[str, Any]) -> tuple[float, dict[str, float]]:
        total, breakdown = 0.0, {}
        for name, weight, params in self.components:
            value = weight * get_reward_component(name)(state, **params)
            breakdown[name] = value
            total += value
        return total, breakdown
