"""Coach v5 statistical layer: uncertainty, detrending, ledger, curve fit, calibration."""

import math

import numpy as np
import pytest

from quadruped_rl.llm_feedback.coach_stats import (
    EffectLedger,
    binomial_se,
    calibration,
    detrended_effect,
    fit_saturating,
    linear_trend,
    move_direction,
    objective_se,
    tracking_ratio,
)


def test_binomial_se_is_floored_at_the_extremes():
    assert binomial_se(0.5, 256) == pytest.approx(math.sqrt(0.25 / 256))
    assert binomial_se(0.0, 256) > 0  # p = 0 still carries uncertainty
    assert binomial_se(0.5, 0) == 0.0


def test_objective_se_combines_rate_and_mean_terms():
    kpi = {"success_rate": 0.5, "mean_forward_velocity_ms": 0.8, "mean_forward_velocity_ms_sd": 0.4}
    w = {"success_rate": 1.0, "mean_forward_velocity_ms": 0.5}
    se = objective_se(kpi, w, 256)
    expected = math.sqrt((math.sqrt(0.25 / 256)) ** 2 + (0.5 * 0.4 / 16) ** 2)
    assert se == pytest.approx(expected)
    # a mean without its sd contributes nothing (unknown), not an error
    assert objective_se({"success_rate": 0.5, "x": 1.0}, {"success_rate": 1, "x": 1}, 100) == (
        pytest.approx(binomial_se(0.5, 100))
    )


def test_detrended_effect_removes_the_learning_trend():
    trace = [(0, 0.0), (10, 0.1), (20, 0.2), (30, 0.3)]  # +0.01 per step
    slope, _ = linear_trend(trace)
    assert slope == pytest.approx(0.01)
    effect, expected = detrended_effect(trace, 40, 0.4)  # rode the trend exactly
    assert expected == pytest.approx(0.1) and effect == pytest.approx(0.0)
    effect, _ = detrended_effect(trace, 40, 0.3)  # J flat although it was rising: harmful
    assert effect == pytest.approx(-0.1)
    assert detrended_effect([(0, 0.5)], 10, 0.7) == (pytest.approx(0.2), 0.0)  # no trend


def test_saturating_fit_recovers_asymptote_and_projects():
    a, tau = 1.5, 8e6
    trace = [(int(t), a - a * math.exp(-t / tau)) for t in np.arange(2e6, 22e6, 2e6)]
    fit = fit_saturating(trace)
    assert fit is not None
    assert fit.asymptote == pytest.approx(a, abs=0.05)
    assert fit.tau_steps == pytest.approx(tau, rel=0.3)
    assert fit.r2 > 0.99
    assert fit.predict(40e6) == pytest.approx(a - a * math.exp(-40e6 / tau), abs=0.05)
    assert fit_saturating(trace[:3]) is None


def test_ledger_shrinks_towards_prior_and_reports_direction():
    assert move_direction(-0.5, -0.3) == "up" and move_direction(1.0, 0.7) == "down"
    led = EffectLedger(prior_sd=0.1)
    mean, sd, n = led.posterior("x", "up")
    assert (mean, sd, n) == (0.0, 0.1, 0)
    led.add("x", "up", 0.2, obs_sd=0.1)  # one observation at prior precision -> halfway
    mean, sd, n = led.posterior("x", "up")
    assert mean == pytest.approx(0.1) and sd == pytest.approx(0.1 / math.sqrt(2)) and n == 1
    for _ in range(20):
        led.add("x", "up", 0.2, obs_sd=0.1)
    assert led.posterior("x", "up")[0] == pytest.approx(0.2, abs=0.01)
    informed = EffectLedger(priors={("t", "up"): (0.05, 0.02)})
    assert informed.posterior("t", "up")[:2] == (pytest.approx(0.05), pytest.approx(0.02))
    assert any("x up" in r and "21 obs" in r for r in led.rows())
    assert len(informed.rows(["t"])) == 2  # prior-only rows when asked for a param


def test_calibration_and_tracking_ratio():
    cal = calibration([(0.1, 0.05), (-0.1, 0.02), (0.0, 0.0), (0.2, 0.3)])
    assert cal["n"] == 4 and cal["sign_accuracy"] == pytest.approx(0.75)
    assert cal["mae"] == pytest.approx((0.05 + 0.12 + 0.0 + 0.1) / 4)
    assert calibration([]) == {"n": 0}
    pairs = [(0.7, 0.6, 0.9), (1.0, 0.9, 0.8), (1.5, 1.3, 0.7), (1.0, 0.1, 0.0)]
    ratio, n = tracking_ratio(pairs)
    assert n == 3 and ratio == pytest.approx((0.42 + 0.9 + 1.95) / (0.49 + 1.0 + 2.25))
    assert tracking_ratio([(1.0, 0.1, 0.0)]) is None
