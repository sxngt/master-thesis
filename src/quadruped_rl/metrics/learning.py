"""Learning-efficiency metrics: convergence speed, sample efficiency, training
stability — computed from logged learning curves (metrics.jsonl / W&B export)."""

from __future__ import annotations

import numpy as np


def samples_to_threshold(steps: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    """First env-step count at which the (smoothed) metric reaches threshold.
    Returns None if never reached — report as censored, not as max."""
    idx = np.argmax(values >= threshold)
    if values[idx] < threshold:
        return None
    return float(steps[idx])


def convergence_step(
    steps: np.ndarray,
    values: np.ndarray,
    window: int = 10,
    rel_tol: float = 0.02,
) -> float | None:
    """Step at which a rolling window's spread falls within rel_tol of the
    final value (plateau detection)."""
    if len(values) < window:
        return None
    final = float(np.mean(values[-window:]))
    if final == 0:
        return None
    for i in range(len(values) - window + 1):
        seg = values[i : i + window]
        if np.all(np.abs(seg - final) <= abs(final) * rel_tol + 1e-9):
            return float(steps[i])
    return None


def training_stability(values: np.ndarray, window: int = 10) -> float:
    """Mean rolling std of the learning curve (lower = more stable training)."""
    if len(values) < window:
        return float(np.std(values))
    rolled = np.lib.stride_tricks.sliding_window_view(values, window)
    return float(np.mean(np.std(rolled, axis=1)))


def area_under_curve(steps: np.ndarray, values: np.ndarray) -> float:
    """Normalized AUC of the learning curve — combined speed+asymptote score."""
    if len(steps) < 2:
        return 0.0
    span = steps[-1] - steps[0]
    return float(np.trapezoid(values, steps) / span) if span > 0 else 0.0
