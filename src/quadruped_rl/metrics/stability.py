"""Stability metrics (KPI group 2)."""

from __future__ import annotations

import numpy as np


def fall_frequency(falls: np.ndarray, duration_s: float) -> float:
    """Falls per minute. `falls` is a per-step binary fall-event array."""
    if duration_s <= 0:
        return 0.0
    return float(np.sum(falls)) / duration_s * 60.0


def attitude_stability(orientations_rpy: np.ndarray) -> float:
    """Combined std of roll and pitch (rad) over an episode [T, 3].
    Lower is more stable."""
    if len(orientations_rpy) < 2:
        return 0.0
    roll_std = float(np.std(orientations_rpy[:, 0]))
    pitch_std = float(np.std(orientations_rpy[:, 1]))
    return float(np.hypot(roll_std, pitch_std))


def contact_force_variance(contact_forces: np.ndarray) -> float:
    """Variance of per-foot ground contact force magnitudes [T, num_feet]."""
    if contact_forces.size == 0:
        return 0.0
    return float(np.var(contact_forces))


def recovery_time_s(
    attitude_deviation: np.ndarray,
    perturbation_step: int,
    dt: float,
    threshold_rad: float = 0.1,
    settle_steps: int = 50,
) -> float | None:
    """Time from an external perturbation until attitude deviation stays
    below threshold for `settle_steps` consecutive steps. None if never."""
    post = attitude_deviation[perturbation_step:]
    below = post < threshold_rad
    run = 0
    for i, ok in enumerate(below):
        run = run + 1 if ok else 0
        if run >= settle_steps:
            return (i - settle_steps + 1) * dt
    return None
