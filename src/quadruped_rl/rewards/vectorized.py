"""Torch-vectorized reward computation for GPU-parallel backends.

Component definitions MUST stay numerically identical to
rewards/traditional.py (the per-state numpy reference) — parity is enforced
by tests/test_rewards.py::test_vectorized_matches_traditional. Weights and
parameters come from the same configs/reward/*.yaml.

State tensors (batched over N envs):
    forward_velocity_ms [N], torques [N, J], orientation_rpy [N, 3],
    foot_slip_velocity [N, F], joint_limit_violation [N, J],
    action_delta [N, J], fallen [N] (bool)
"""

from __future__ import annotations

from typing import Any

import torch


def _forward_velocity(s: dict, target_ms: float = 1.0, **_) -> torch.Tensor:
    v = s["forward_velocity_ms"]
    return torch.exp(-4.0 * (v - target_ms) ** 2)


def _energy(s: dict, **_) -> torch.Tensor:
    return torch.sum(torch.square(s["torques"]), dim=-1)


def _stability(s: dict, **_) -> torch.Tensor:
    rpy = s["orientation_rpy"]
    return rpy[:, 0] ** 2 + rpy[:, 1] ** 2


def _foot_slip(s: dict, **_) -> torch.Tensor:
    return torch.sum(torch.square(s["foot_slip_velocity"]), dim=-1)


def _joint_limit(s: dict, **_) -> torch.Tensor:
    return torch.sum(torch.clamp(s["joint_limit_violation"], min=0.0), dim=-1)


def _action_rate(s: dict, **_) -> torch.Tensor:
    return torch.sum(torch.square(s["action_delta"]), dim=-1)


def _alive_bonus(s: dict, **_) -> torch.Tensor:
    return (~s["fallen"]).float()


def _termination(s: dict, **_) -> torch.Tensor:
    return s["fallen"].float()


_COMPONENTS = {
    "forward_velocity": _forward_velocity,
    "energy": _energy,
    "stability": _stability,
    "foot_slip": _foot_slip,
    "joint_limit": _joint_limit,
    "action_rate": _action_rate,
    "alive_bonus": _alive_bonus,
    "termination": _termination,
}


class VectorizedTraditionalReward:
    """Batched equivalent of rewards.traditional.TraditionalReward."""

    def __init__(self, reward_cfg: dict[str, Any]):
        self.components: list[tuple[str, float, dict]] = []
        for name, params in reward_cfg["components"].items():
            params = dict(params)
            weight = params.pop("weight")
            if name not in _COMPONENTS:
                raise KeyError(f"No vectorized implementation for component '{name}'")
            self.components.append((name, weight, params))

    def __call__(
        self, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total: torch.Tensor | None = None
        breakdown: dict[str, torch.Tensor] = {}
        for name, weight, params in self.components:
            value = weight * _COMPONENTS[name](state, **params)
            breakdown[name] = value
            total = value if total is None else total + value
        assert total is not None
        return total, breakdown
