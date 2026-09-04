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


def _lateral_velocity(s: dict, **_) -> torch.Tensor:
    return torch.square(s["lateral_velocity_ms"])


def _yaw_rate(s: dict, **_) -> torch.Tensor:
    return torch.square(s["yaw_rate_rads"])


def _feet_air_time(s: dict, target_s: float = 0.5, **_) -> torch.Tensor:
    moving = (s.get("command_speed", torch.ones_like(s["forward_velocity_ms"])) > 0.1).float()
    return (
        torch.sum((s["feet_last_air_time"] - target_s) * s["feet_first_contact"], dim=-1) * moving
    )


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
    "lateral_velocity": _lateral_velocity,
    "yaw_rate": _yaw_rate,
    "feet_air_time": _feet_air_time,
    "alive_bonus": _alive_bonus,
    "termination": _termination,
}


class VectorizedTraditionalReward:
    """Batched equivalent of rewards.traditional.TraditionalReward.

    Parameters (weights and per-component params) are mutable at runtime via
    set_params() so a reward scheduler (llm_feedback/coach.py) can reshape the
    reward mid-training. Per-component running sums are accumulated on-device
    (no host sync) and harvested with pop_stats() for the coach report.
    """

    def __init__(self, reward_cfg: dict[str, Any]):
        self.components: list[tuple[str, float, dict]] = []
        for name, params in reward_cfg["components"].items():
            params = dict(params)
            weight = params.pop("weight")
            if name not in _COMPONENTS:
                raise KeyError(f"No vectorized implementation for component '{name}'")
            self.components.append((name, float(weight), params))
        self._sums: dict[str, torch.Tensor] = {}
        self._count = 0

    # ------------------------------------------------------------ params
    def get_params(self) -> dict[str, float]:
        """Flat view: {'energy.weight': -2.5e-5, 'feet_air_time.target_s': 0.5, ...}."""
        flat: dict[str, float] = {}
        for name, weight, params in self.components:
            flat[f"{name}.weight"] = weight
            for k, v in params.items():
                flat[f"{name}.{k}"] = float(v)
        return flat

    def set_params(self, updates: dict[str, float]) -> None:
        """Apply flat-key updates in place; unknown keys raise KeyError."""
        index = {name: i for i, (name, _, _) in enumerate(self.components)}
        for key, value in updates.items():
            comp, _, field = key.partition(".")
            if comp not in index or not field:
                raise KeyError(f"Unknown reward parameter '{key}'")
            name, weight, params = self.components[index[comp]]
            if field == "weight":
                self.components[index[comp]] = (name, float(value), params)
            elif field in params:
                params[field] = float(value)
            else:
                raise KeyError(f"Unknown reward parameter '{key}'")

    # ------------------------------------------------------------- stats
    def pop_stats(self) -> dict[str, float]:
        """Mean per-step contribution of each weighted component since the
        last call (plus 'total'). Empty dict if nothing was accumulated."""
        if self._count == 0:
            return {}
        out = {k: float(v) / self._count for k, v in self._sums.items()}
        out["total"] = sum(out.values())
        self._sums, self._count = {}, 0
        return out

    def __call__(
        self, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total: torch.Tensor | None = None
        breakdown: dict[str, torch.Tensor] = {}
        for name, weight, params in self.components:
            value = weight * _COMPONENTS[name](state, **params)
            breakdown[name] = value
            total = value if total is None else total + value
            s = value.detach().sum()
            self._sums[name] = s if name not in self._sums else self._sums[name] + s
        assert total is not None
        self._count += int(total.shape[0])
        return total, breakdown
