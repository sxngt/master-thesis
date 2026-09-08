"""Pure-function tests for the coach log study (no run artefacts needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quadruped_rl.analysis.coach_study import (
    action_table,
    cost_table,
    counterfactual_objective,
    tracking_fit,
)


def _calls() -> pd.DataFrame:
    rows = []
    for i, (t, v, s, ok) in enumerate(
        [
            (1.0, 0.9, 0.8, True),
            (0.7, 0.6, 0.6, True),
            (1.5, 1.3, 0.9, True),
            (1.0, 0.1, 0.0, True),
            (1.0, 0.5, 0.9, False),
        ]
    ):
        rows.append(
            {
                "batch": "v4",
                "run_id": "r",
                "setting": "s",
                "seed": 0,
                "k": i + 1,
                "status": "kept",
                "applied": {"forward_velocity.target_ms": t} if i < 2 else {"energy.weight": -1e-5},
                "objective_before": 0.5,
                "objective_after": 0.6,
                "target_before": t,
                "velocity": v,
                "success": s,
                "params_consistent": ok,
                "input_tokens": 1000,
                "output_tokens": 2000,
                "reasoning_tokens": 1500,
            }
        )
    return pd.DataFrame(rows)


def test_tracking_fit_uses_only_consistent_established_rows():
    fit = tracking_fit(_calls(), min_success=0.5)
    assert fit["n"] == 3  # success 0.0 row and inconsistent row excluded
    assert fit["ratio_mean"] == pytest.approx((0.9 + 0.6 / 0.7 + 1.3 / 1.5) / 3)
    assert 0.85 < fit["slope_through_origin"] < 0.9


def test_counterfactual_keeps_success_and_rescales_velocity():
    cf = counterfactual_objective(np.array([0.5, 0.7]), np.array([1.0]), ratio=0.9)
    # J = success + 0.5 * 0.9 * 1.0 -> mean of 0.95 and 1.15
    assert cf.loc[0, "mean"] == pytest.approx(1.05)
    assert cf.loc[0, "n"] == 2 and cf.loc[0, "ci_lo"] < 1.05 < cf.loc[0, "ci_hi"]


def test_cost_table_prices_tokens():
    cost = cost_table(_calls(), usd_per_m_in=2.5, usd_per_m_out=15.0, krw_per_usd=1000.0)
    assert cost.loc[0, "usd"] == pytest.approx(5 * (1000 / 1e6 * 2.5 + 2000 / 1e6 * 15))
    assert cost.loc[0, "krw"] == pytest.approx(cost.loc[0, "usd"] * 1000)


def test_action_table_labels_target_direction():
    a = action_table(_calls())
    assert list(a["direction"][:2]) == ["target up/keep", "target down"]
    assert (a["direction"][2:] == "other").all()
    assert a["delta_j"].iloc[0] == pytest.approx(0.1)
