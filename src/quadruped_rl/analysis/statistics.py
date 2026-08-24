"""Statistical analysis for algorithm comparison (thesis 1.2.2).

Pipeline per metric: one-way/two-way ANOVA -> Tukey HSD post-hoc ->
Cohen's d effect sizes -> 95% CIs. Every thesis comparison table is
produced by compare_algorithms() so methodology stays uniform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def confidence_interval(x: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """(mean, ci_low, ci_high) via t-distribution."""
    x = np.asarray(x, float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, mean, mean
    sem = stats.sem(x)
    margin = sem * stats.t.ppf((1 + confidence) / 2, len(x) - 1)
    return mean, mean - margin, mean + margin


def one_way_anova(groups: dict[str, np.ndarray]) -> dict[str, float]:
    """groups: {algorithm_name: metric_values_across_seeds}."""
    f_stat, p_value = stats.f_oneway(*groups.values())
    # eta-squared effect size
    all_vals = np.concatenate(list(groups.values()))
    grand = all_vals.mean()
    ss_between = sum(len(v) * (np.mean(v) - grand) ** 2 for v in groups.values())
    ss_total = float(np.sum((all_vals - grand) ** 2))
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
    return {"f_stat": float(f_stat), "p_value": float(p_value), "eta_squared": float(eta_sq)}


def tukey_hsd(groups: dict[str, np.ndarray], alpha: float = 0.05) -> pd.DataFrame:
    """Pairwise Tukey HSD. Returns tidy DataFrame with reject flags + Cohen's d."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    values = np.concatenate(list(groups.values()))
    labels = np.concatenate([[k] * len(v) for k, v in groups.items()])
    res = pairwise_tukeyhsd(values, labels, alpha=alpha)
    df = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    df["cohens_d"] = [
        cohens_d(groups[row["group1"]], groups[row["group2"]]) for _, row in df.iterrows()
    ]
    return df


def compare_algorithms(
    results: pd.DataFrame, metric: str, group_col: str = "algorithm", alpha: float = 0.05
) -> dict[str, Any]:
    """Full comparison for one metric.

    results: long-format DataFrame with columns [algorithm, robot, terrain,
    seed, <metric>]. Returns ANOVA + Tukey + per-group summary (mean, std, CI).
    """
    groups = {name: g[metric].to_numpy() for name, g in results.groupby(group_col)}
    if any(len(v) < 2 for v in groups.values()):
        raise ValueError(
            "Every group needs >= 2 samples (seeds) — "
            "never report single-seed comparisons (CLAUDE.md)."
        )
    summary = {}
    for name, vals in groups.items():
        mean, lo, hi = confidence_interval(vals)
        summary[name] = {
            "n": len(vals),
            "mean": mean,
            "std": float(np.std(vals, ddof=1)),
            "ci95": (lo, hi),
        }
    return {
        "metric": metric,
        "anova": one_way_anova(groups),
        "tukey": tukey_hsd(groups, alpha),
        "summary": summary,
    }


def load_results_table(results_root: str, experiment: str) -> pd.DataFrame:
    """Assemble a long-format results table from per-run config.yaml +
    final eval metrics under data/results/."""
    import json
    from pathlib import Path

    import yaml

    rows = []
    for run_dir in Path(results_root).iterdir():
        cfg_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.jsonl"
        if not (cfg_path.exists() and metrics_path.exists()):
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        if cfg.get("experiment", {}).get("name") != experiment:
            continue
        final = {}
        for line in metrics_path.read_text().splitlines():
            rec = json.loads(line)
            fin = {k.removeprefix("final/"): v for k, v in rec.items() if k.startswith("final/")}
            if fin:
                final = fin
        if final:
            rows.append(
                {
                    "algorithm": cfg["algorithm"]["name"],
                    "robot": cfg["robot"]["name"],
                    "terrain": cfg["terrain"]["name"],
                    "seed": cfg["run"]["seed"],
                    **final,
                }
            )
    return pd.DataFrame(rows)
