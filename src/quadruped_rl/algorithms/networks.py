"""Shared actor/critic network builders (torch).

Architectures per thesis 1.1.2: actor MLP 512-256-128 (or LSTM variant),
critic MLP 512-512-256-128. Activation / normalization configurable for the
comparison experiments (relu | tanh | elu | swish; layer/spectral norm).
"""

from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "swish": nn.SiLU}


def mlp(
    in_dim: int,
    hidden: list[int],
    out_dim: int | None,
    activation: str = "relu",
    layer_norm: bool = False,
    spectral_norm: bool = False,
) -> nn.Sequential:
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        lin = nn.Linear(prev, h)
        if spectral_norm:
            lin = nn.utils.parametrizations.spectral_norm(lin)
        layers.append(lin)
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    """Diagonal Gaussian policy with state-independent log-std."""

    def __init__(self, obs_dim: int, act_dim: int, spec: dict):
        super().__init__()
        self.body = mlp(
            obs_dim,
            spec["hidden"],
            act_dim,
            activation=spec.get("activation", "relu"),
            layer_norm=spec.get("layer_norm", False),
            spectral_norm=spec.get("spectral_norm", False),
        )
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.body(obs), self.log_std.exp())


class Critic(nn.Module):
    """State-value critic V(s)."""

    def __init__(self, obs_dim: int, spec: dict):
        super().__init__()
        self.net = mlp(obs_dim, spec["hidden"], 1, activation=spec.get("activation", "relu"))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class QCritic(nn.Module):
    """State-action critic Q(s, a) for off-policy algorithms."""

    def __init__(self, obs_dim: int, act_dim: int, spec: dict):
        super().__init__()
        self.net = mlp(
            obs_dim + act_dim, spec["hidden"], 1, activation=spec.get("activation", "relu")
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class DeterministicActor(nn.Module):
    """Deterministic policy mu(s) with tanh-bounded actions in [-1, 1]
    (TD3/DDPG)."""

    def __init__(self, obs_dim: int, act_dim: int, spec: dict):
        super().__init__()
        self.net = mlp(
            obs_dim,
            spec["hidden"],
            act_dim,
            activation=spec.get("activation", "relu"),
            layer_norm=spec.get("layer_norm", False),
            spectral_norm=spec.get("spectral_norm", False),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))
