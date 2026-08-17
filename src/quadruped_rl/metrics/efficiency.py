"""Efficiency metrics (KPI group 3)."""

from __future__ import annotations

import numpy as np

GRAVITY = 9.81


def cost_of_transport(energy_j: float, mass_kg: float, distance_m: float) -> float:
    """CoT = E / (m * g * d). Dimensionless; lower is better.
    This exact definition is fixed thesis-wide (docs/metrics.md)."""
    if distance_m <= 0 or mass_kg <= 0:
        return float("inf")
    return energy_j / (mass_kg * GRAVITY * distance_m)


def torque_efficiency(torques: np.ndarray, distance_m: float) -> float:
    """RMS joint torque per meter traveled. torques: [T, num_joints]."""
    if distance_m <= 0 or torques.size == 0:
        return float("inf")
    rms = float(np.sqrt(np.mean(np.square(torques))))
    return rms / distance_m


def mechanical_power(torques: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
    """Instantaneous positive mechanical power per step: sum |tau * qdot|."""
    return np.sum(np.abs(torques * joint_velocities), axis=-1)
