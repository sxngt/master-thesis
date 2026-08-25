"""KPI implementations vs. docs/metrics.md definitions."""

import numpy as np

from quadruped_rl.metrics.efficiency import cost_of_transport, torque_efficiency
from quadruped_rl.metrics.learning import (
    area_under_curve,
    convergence_step,
    samples_to_threshold,
    training_stability,
)
from quadruped_rl.metrics.locomotion import (
    completion_time_s,
    mean_forward_velocity,
    path_efficiency,
    success_rate,
)
from quadruped_rl.metrics.stability import (
    attitude_stability,
    fall_frequency,
    recovery_time_s,
)


def test_mean_forward_velocity_straight_line():
    # 1 m/s along x for 10 steps at dt=0.1
    pos = np.column_stack([np.linspace(0, 1.0, 11), np.zeros(11), np.zeros(11)])
    assert np.isclose(mean_forward_velocity(pos, 0.1), 1.0)


def test_path_efficiency_bounds():
    straight = np.column_stack([np.linspace(0, 5, 50), np.zeros(50), np.zeros(50)])
    assert np.isclose(path_efficiency(straight, 5.0), 1.0)
    zigzag = np.column_stack([np.linspace(0, 5, 50), np.tile([0, 1], 25), np.zeros(50)])
    assert path_efficiency(zigzag, 5.0) < 1.0


def test_success_rate():
    assert success_rate([True, True, False, False]) == 0.5
    assert success_rate([]) == 0.0


def test_completion_time():
    assert completion_time_s(500, 0.02) == 10.0


def test_cost_of_transport_definition():
    # CoT = E / (m g d): 100 J, 10 kg, 1 m -> 100/(10*9.81*1)
    assert np.isclose(cost_of_transport(100.0, 10.0, 1.0), 100.0 / 98.1)
    assert cost_of_transport(1.0, 10.0, 0.0) == float("inf")


def test_torque_efficiency():
    tau = np.full((100, 12), 2.0)
    assert np.isclose(torque_efficiency(tau, 4.0), 0.5)


def test_fall_frequency_per_minute():
    falls = np.zeros(600)
    falls[[10, 300]] = 1
    assert np.isclose(fall_frequency(falls, 60.0), 2.0)


def test_attitude_stability_flat_is_zero():
    rpy = np.zeros((100, 3))
    assert attitude_stability(rpy) == 0.0


def test_recovery_time():
    dev = np.concatenate([np.full(10, 0.5), np.full(100, 0.05)])
    t = recovery_time_s(dev, perturbation_step=0, dt=0.02, threshold_rad=0.1, settle_steps=50)
    assert t is not None and np.isclose(t, 10 * 0.02)
    never = recovery_time_s(np.full(100, 0.5), 0, 0.02)
    assert never is None


def test_learning_metrics():
    steps = np.arange(0, 1000, 100, dtype=float)
    values = np.minimum(steps / 500.0, 1.0)
    assert samples_to_threshold(steps, values, 0.99) == 500.0
    assert samples_to_threshold(steps, values, 2.0) is None
    assert convergence_step(steps, values, window=3) is not None
    assert training_stability(np.ones(50)) == 0.0
    assert area_under_curve(steps, np.ones_like(steps)) == 1.0
