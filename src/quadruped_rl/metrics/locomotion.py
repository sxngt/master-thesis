"""Locomotion performance metrics (KPI group 1)."""

from __future__ import annotations

import numpy as np


def mean_forward_velocity(positions: np.ndarray, dt: float) -> float:
    """Average forward speed (m/s) from base positions [T, 3] sampled at dt."""
    if len(positions) < 2:
        return 0.0
    displacement = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    duration = (len(positions) - 1) * dt
    return displacement / duration if duration > 0 else 0.0


def path_efficiency(positions: np.ndarray, straight_line_m: float) -> float:
    """Straight-line distance / actual traveled path length, in (0, 1]."""
    steps = np.diff(positions[:, :2], axis=0)
    traveled = float(np.sum(np.linalg.norm(steps, axis=1)))
    if traveled <= 0:
        return 0.0
    if straight_line_m <= 0:
        straight_line_m = float(np.linalg.norm(positions[-1, :2] - positions[0, :2]))
    return min(straight_line_m / traveled, 1.0)


def success_rate(reached: list[bool] | np.ndarray) -> float:
    """Fraction of episodes that reached the goal."""
    reached = np.asarray(reached, dtype=float)
    return float(reached.mean()) if reached.size else 0.0


def completion_time_s(steps: int, dt: float) -> float:
    """Wall-clock episode duration in seconds."""
    return steps * dt
