"""Statistical analysis: ANOVA, Tukey, Cohen's d, CIs."""

import numpy as np
import pandas as pd
import pytest

from quadruped_rl.analysis.statistics import (
    cohens_d,
    compare_algorithms,
    confidence_interval,
    one_way_anova,
    tukey_hsd,
)

RNG = np.random.default_rng(0)


def test_cohens_d_known_case():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + 2.0
    d = cohens_d(b, a)
    assert np.isclose(d, 2.0 / a.std(ddof=1))


def test_confidence_interval_contains_mean():
    x = RNG.normal(5.0, 1.0, 30)
    mean, lo, hi = confidence_interval(x)
    assert lo < mean < hi


def test_anova_detects_difference():
    groups = {
        "ppo": RNG.normal(0.9, 0.02, 10),
        "sac": RNG.normal(0.7, 0.02, 10),
        "ddpg": RNG.normal(0.5, 0.02, 10),
    }
    res = one_way_anova(groups)
    assert res["p_value"] < 0.001
    assert res["eta_squared"] > 0.5
    tukey = tukey_hsd(groups)
    assert len(tukey) == 3  # 3 pairwise comparisons
    assert "cohens_d" in tukey.columns


def test_compare_algorithms_rejects_single_seed():
    df = pd.DataFrame({"algorithm": ["ppo", "sac"], "success_rate": [0.9, 0.8]})
    with pytest.raises(ValueError, match="single-seed"):
        compare_algorithms(df, "success_rate")


def test_compare_algorithms_full():
    rows = []
    for algo, mu in [("ppo", 0.9), ("sac", 0.8)]:
        for seed in range(10):
            rows.append(
                {"algorithm": algo, "seed": seed, "success_rate": float(RNG.normal(mu, 0.03))}
            )
    report = compare_algorithms(pd.DataFrame(rows), "success_rate")
    assert set(report["summary"]) == {"ppo", "sac"}
    assert report["anova"]["p_value"] < 0.05
