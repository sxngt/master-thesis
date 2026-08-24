"""Learned human-preference reward model (Bradley-Terry over trajectory pairs).

Trained on PreferencePair labels
provides the R_human_preference term of the
hybrid reward. Updated periodically during training
(configs/reward/hybrid_llm.yaml: update_interval_steps).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from quadruped_rl.algorithms.networks import mlp


class PreferenceRewardModel:
    def __init__(
        self,
        feature_dim: int,
        hidden: list[int] | None = None,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.device = device
        self.net = mlp(feature_dim, hidden or [256, 256], 1, activation="relu").to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def score_state(self, state: dict[str, Any]) -> float:
        feats = torch.as_tensor(
            state["preference_features"], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            return float(self.net(feats.unsqueeze(0)).squeeze())

    def _traj_return(self, features: torch.Tensor) -> torch.Tensor:
        """Sum of per-step predicted rewards over a trajectory [T, F]."""
        return self.net(features).sum()

    def train_on_pairs(
        self, pairs: list[tuple[np.ndarray, np.ndarray, float]], epochs: int = 5
    ) -> dict[str, float]:
        """pairs: (features_a [T,F], features_b [T,F], label) where label is
        1.0 if a preferred, 0.0 if b preferred, 0.5 if equal (Bradley-Terry)."""
        losses = []
        for _ in range(epochs):
            for fa, fb, label in pairs:
                ra = self._traj_return(torch.as_tensor(fa, dtype=torch.float32, device=self.device))
                rb = self._traj_return(torch.as_tensor(fb, dtype=torch.float32, device=self.device))
                logit = ra - rb
                loss = F.binary_cross_entropy_with_logits(
                    logit, torch.tensor(label, device=self.device)
                )
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(float(loss))
        return {"pref_loss": float(np.mean(losses)) if losses else 0.0}

    def save(self, path: str | Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
