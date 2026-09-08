"""Log study of the LLM reward coach across batches (Phase 3 pilot).

Everything here is derived from the recorded artefacts of finished runs —
``coach_log.jsonl`` (one record per coach report: prompt, response, applied
parameters, KPI before/after, token usage) and ``metrics.jsonl`` — so no new
training or API calls are needed. Two kinds of output are produced and must
be kept apart in the thesis:

* **measured**: learning curves, J decomposition, intervention outcomes,
  parameter usage, token cost — facts about runs that happened;
* **estimated**: the counterfactual of ``counterfactual_objective``, which
  re-scores measured success rates under an assumed commanded speed using the
  empirically fitted command-tracking ratio. It is a model prediction with
  stated assumptions, never a result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quadruped_rl.analysis.coach import DEFAULT_OBJECTIVE, _read_jsonl, load_coach_table
from quadruped_rl.analysis.plots import STYLE, _save
from quadruped_rl.analysis.statistics import confidence_interval

TARGET_KEY = "forward_velocity.target_ms"
VEL_KEY = "mean_forward_velocity_ms"
COLORS = {
    "none": "#555555",
    "random": "#999999",
    "hillclimb": "#1f77b4",
    "llm": "#d62728",
    "llm_v3": "#ff9896",
    "llm_v4": "#d62728",
}


# ------------------------------------------------------------------ calls
def collect_calls(batch_roots: dict[str, str | Path]) -> pd.DataFrame:
    """One row per settled coach report of every LLM run under the given batch roots.

    ``batch_roots`` maps a batch label (e.g. ``"v3"``) to its ``runs`` dir.
    Pending lines are merged with their settled line (kept | rolled_back |
    noop) so each (run, k) appears once, with token usage attached.
    """
    rows: list[dict[str, Any]] = []
    for batch, root in batch_roots.items():
        for run_dir in sorted(Path(root).iterdir()):
            if "coach-llm" not in run_dir.name:
                continue
            log = _read_jsonl(run_dir / "coach_log.jsonl")
            if not log:
                continue
            by_k: dict[int, dict[str, Any]] = {}
            for rec in log:
                cur = by_k.setdefault(rec["k"], dict(rec))
                if rec["status"] != "pending":
                    cur.update({k: v for k, v in rec.items() if k != "usage"})
                if rec.get("usage"):
                    cur["usage"] = rec["usage"]
            prev_status = None
            for k in sorted(by_k):
                r = by_k[k]
                usage = r.get("usage") or {}
                applied = r.get("applied") or {}
                pb = r.get("params_before") or {}
                kpi = r.get("kpi_before") or {}
                diag = str(r.get("diagnosis") or "")
                rows.append(
                    {
                        "batch": batch,
                        "run_id": run_dir.name,
                        "setting": _setting(run_dir),
                        "seed": _seed(run_dir),
                        "k": k,
                        "step": r["step"],
                        "status": r["status"],
                        "restored": r.get("restored_from_step") is not None,
                        "api_error": diag.startswith("api error"),
                        "discarded": diag.startswith("discarded"),
                        "n_applied": len(applied),
                        "applied": applied,
                        "target_before": pb.get(TARGET_KEY),
                        "target_after": applied.get(TARGET_KEY, pb.get(TARGET_KEY)),
                        "success": kpi.get("success_rate"),
                        "velocity": kpi.get(VEL_KEY),
                        "objective_before": r.get("objective_before"),
                        "objective_after": r.get("objective_after"),
                        "tolerance_used": r.get("tolerance_used"),
                        "confidence": r.get("confidence"),
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "reasoning_tokens": usage.get("reasoning_tokens", 0),
                        # the KPI of this report was measured under the parameters in
                        # force since the previous report — unless that intervention
                        # was rolled back, in which case params_before is already the
                        # restored set and the pair (params, KPI) is inconsistent
                        "params_consistent": prev_status != "rolled_back"
                        and r.get("restored_from_step") is None,
                    }
                )
                prev_status = r["status"]
    return pd.DataFrame(rows)


def _setting(run_dir: Path) -> str:
    import yaml

    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return (
        f"{cfg['terrain']['name']}-{cfg['sim'].get('terrain_level', 'easy')}-"
        f"{cfg.get('reward', {}).get('name', 'traditional')}"
    )


def _seed(run_dir: Path) -> int:
    import yaml

    return int(yaml.safe_load((run_dir / "config.yaml").read_text())["run"]["seed"])


# ------------------------------------------------------------ derived tables
def action_table(calls: pd.DataFrame) -> pd.DataFrame:
    """One row per applied parameter change: parameter, direction, outcome, delta J."""
    rows = []
    for _, c in calls.iterrows():
        for key, val in (c["applied"] or {}).items():
            rows.append(
                {
                    "batch": c["batch"],
                    "setting": c["setting"],
                    "seed": c["seed"],
                    "k": c["k"],
                    "param": key,
                    "value": float(val),
                    "status": c["status"],
                    "delta_j": (c["objective_after"] - c["objective_before"])
                    if c["objective_after"] is not None and c["objective_before"] is not None
                    else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["direction"] = np.where(
        df["param"] == TARGET_KEY,
        np.where(df["value"] >= 1.0, "target up/keep", "target down"),
        "other",
    )
    return df


def tracking_fit(calls: pd.DataFrame, min_success: float = 0.5) -> dict[str, float]:
    """Command-tracking ratio v / target from reports with an established gait.

    Uses only rows whose (parameters, KPI) pair is consistent and whose success
    rate is at least ``min_success`` — before that the policy does not track
    its command at all (ratio ~0.1-0.3) and the ratio is meaningless.
    """
    d = calls[calls["params_consistent"] & (calls["success"] >= min_success)]
    d = d.dropna(subset=["target_before", "velocity"])
    if len(d) < 3:
        return {"n": len(d), "ratio_mean": float("nan"), "ratio_sd": float("nan")}
    ratio = d["velocity"] / d["target_before"]
    slope = float(np.sum(d["velocity"] * d["target_before"]) / np.sum(d["target_before"] ** 2))
    return {
        "n": int(len(d)),
        "ratio_mean": float(ratio.mean()),
        "ratio_sd": float(ratio.std(ddof=1)),
        "slope_through_origin": slope,
        "target_min": float(d["target_before"].min()),
        "target_max": float(d["target_before"].max()),
    }


def counterfactual_objective(
    success: np.ndarray,
    targets: np.ndarray,
    ratio: float,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """ESTIMATE: J each seed would score if its commanded speed were ``target``.

    J_cf = w_s * success_seed + w_v * ratio * target, i.e. the measured
    success rate is kept and the velocity term is re-scored with the fitted
    command-tracking ratio. Assumptions (stated with every use): success is
    insensitive to the commanded speed over the target range, and the policy
    would track a raised command with the same ratio it tracks the current
    one. Returns mean, sd and 95 % CI over seeds per target.
    """
    w = weights or DEFAULT_OBJECTIVE
    rows = []
    for t in np.atleast_1d(targets):
        j = w["success_rate"] * np.asarray(success) + w[VEL_KEY] * ratio * float(t)
        mean, lo, hi = confidence_interval(j)
        rows.append(
            {
                "target_ms": float(t),
                "n": len(j),
                "mean": mean,
                "std": float(np.std(j, ddof=1)) if len(j) > 1 else 0.0,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame(rows)


def cost_table(
    calls: pd.DataFrame, usd_per_m_in: float, usd_per_m_out: float, krw_per_usd: float
) -> pd.DataFrame:
    g = calls.groupby("batch").agg(
        runs=("run_id", "nunique"),
        calls=("k", "size"),
        input_tokens=("input_tokens", "sum"),
        output_tokens=("output_tokens", "sum"),
        reasoning_tokens=("reasoning_tokens", "sum"),
    )
    g["usd"] = g["input_tokens"] / 1e6 * usd_per_m_in + g["output_tokens"] / 1e6 * usd_per_m_out
    g["krw"] = g["usd"] * krw_per_usd
    g["usd_per_run"] = g["usd"] / g["runs"]
    return g.reset_index()


def outcome_table(calls: pd.DataFrame) -> pd.DataFrame:
    """Per (batch, setting): mean per run of kept / rolled back / restored / noop / api error."""
    c = calls.copy()
    c["kept"] = c["status"] == "kept"
    c["rolled_back"] = c["status"] == "rolled_back"
    c["noop"] = c["status"] == "noop"
    per_run = c.groupby(["batch", "setting", "run_id"]).agg(
        reports=("k", "size"),
        kept=("kept", "sum"),
        rolled_back=("rolled_back", "sum"),
        restored=("restored", "sum"),
        noop=("noop", "sum"),
        api_errors=("api_error", "sum"),
    )
    return per_run.groupby(["batch", "setting"]).mean().round(2).reset_index()


def decomposition(table: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Final J split into its terms per (setting, condition), mean over seeds."""
    w = weights or DEFAULT_OBJECTIVE
    d = table.copy()
    d["term_success"] = w["success_rate"] * d["success_rate"]
    d["term_velocity"] = w[VEL_KEY] * d[VEL_KEY]
    return (
        d.groupby(["setting", "condition"])
        .agg(
            n=("seed", "size"),
            success=("success_rate", "mean"),
            velocity=(VEL_KEY, "mean"),
            term_success=("term_success", "mean"),
            term_velocity=("term_velocity", "mean"),
            objective=("objective_final", "mean"),
            objective_sd=("objective_final", "std"),
        )
        .reset_index()
    )


# ------------------------------------------------------------------ figures
def _band(ax, table: pd.DataFrame, label: str, color: str, grid: np.ndarray) -> None:
    curves = []
    for _, r in table.iterrows():
        s, o = r["_steps"], r["_objective"]
        if len(s) < 2:
            continue
        curves.append(np.interp(grid, s, o, left=np.nan, right=np.nan))
    if not curves:
        return
    c = np.array(curves)
    mean = np.nanmean(c, axis=0)
    n = np.sum(~np.isnan(c), axis=0)
    sd = np.nanstd(c, axis=0, ddof=1) if len(c) > 1 else np.zeros_like(mean)
    half = 1.96 * sd / np.sqrt(np.maximum(n, 1))
    ax.plot(grid / 1e6, mean, color=color, label=f"{label} (n={len(c)})")
    ax.fill_between(grid / 1e6, mean - half, mean + half, color=color, alpha=0.15, lw=0)


def objective_curves_by_batch(
    table: pd.DataFrame, setting: str, out: str | Path, groups: list[tuple[str, str, str]]
) -> None:
    """Mean J ± 95 % CI over seeds; ``groups`` = [(label, condition, batch)]."""
    d = table[table["setting"] == setting]
    grid = np.linspace(0, d["_steps"].map(lambda s: s[-1] if len(s) else 0).max(), 81)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        for label, cond, batch in groups:
            sub = d[(d["condition"] == cond) & (d["batch"] == batch)]
            if len(sub):
                _band(ax, sub, label, COLORS.get(f"{cond}_{batch}", COLORS[cond]), grid)
        ax.set_xlabel("training steps (M)")
        ax.set_ylabel("objective J")
        ax.set_title(setting)
        ax.legend(fontsize=7)
        _save(fig, out)


def target_timelines(calls: pd.DataFrame, setting: str, out: str | Path) -> None:
    """Commanded speed over training for every LLM run of a setting (v3 vs v4)."""
    d = calls[calls["setting"] == setting]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0), sharey=True)
        for ax, batch in zip(axes, ["v3", "v4"], strict=True):
            for _, run in d[d["batch"] == batch].groupby("run_id"):
                run = run.sort_values("k")
                steps = [0.0] + list(run["step"] / 1e6)
                vals = [1.0] + list(run["target_after"].astype(float))
                ax.step(steps, vals, where="post", alpha=0.6, lw=1.2)
            ax.axhline(1.0, color="k", ls=":", lw=0.8, label="baseline command (none)")
            ax.set_title(f"{setting} — LLM {batch}")
            ax.set_xlabel("training steps (M)")
        axes[0].set_ylabel("forward_velocity.target_ms")
        axes[0].legend(fontsize=7)
        _save(fig, out)


def decomposition_bars(dec: pd.DataFrame, out: str | Path, order: list[str]) -> None:
    settings = list(dec["setting"].unique())
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, len(settings), figsize=(2.6 * len(settings), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, setting in zip(axes, settings, strict=True):
            d = dec[dec["setting"] == setting].set_index("condition").reindex(order).dropna()
            x = np.arange(len(d))
            ax.bar(x, d["term_success"], color="#4c72b0", label="success term")
            ax.bar(
                x,
                d["term_velocity"],
                bottom=d["term_success"],
                color="#dd8452",
                label="0.5 x velocity term",
            )
            ax.errorbar(x, d["objective"], yerr=d["objective_sd"], fmt="none", ecolor="k", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{c}\n(n={int(n)})" for c, n in zip(d.index, d["n"], strict=True)], fontsize=7
            )
            ax.set_title(setting, fontsize=8)
        axes[0].set_ylabel("final objective J")
        axes[0].legend(fontsize=7)
        _save(fig, out)


def tracking_scatter(calls: pd.DataFrame, fit: dict[str, float], out: str | Path) -> None:
    d = calls[calls["params_consistent"]].dropna(subset=["target_before", "velocity"])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.3))
        low = d[d["success"] < 0.5]
        high = d[d["success"] >= 0.5]
        ax.scatter(
            low["target_before"], low["velocity"], s=8, color="#bbbbbb", label="success < 0.5"
        )
        ax.scatter(
            high["target_before"], high["velocity"], s=10, color="#d62728", label="success >= 0.5"
        )
        if np.isfinite(fit.get("slope_through_origin", np.nan)):
            xs = np.linspace(0.3, 1.6, 10)
            ax.plot(
                xs,
                fit["slope_through_origin"] * xs,
                "k--",
                lw=1,
                label=f"v = {fit['slope_through_origin']:.2f} x target",
            )
        ax.set_xlabel("commanded speed target_ms (m/s)")
        ax.set_ylabel("measured velocity (m/s)")
        ax.legend(fontsize=7)
        _save(fig, out)


def counterfactual_plot(
    cf: pd.DataFrame, none_mean: float, none_ci: tuple[float, float], out: str | Path
) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        ax.plot(
            cf["target_ms"], cf["mean"], color="#d62728", label="LLM v4 seeds re-scored (estimate)"
        )
        ax.fill_between(
            cf["target_ms"], cf["ci_lo"], cf["ci_hi"], color="#d62728", alpha=0.15, lw=0
        )
        ax.axhline(none_mean, color="#555555", label="none, measured (n=10)")
        ax.axhspan(none_ci[0], none_ci[1], color="#555555", alpha=0.12, lw=0)
        ax.axvline(1.0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("commanded speed the coach would restore to (m/s)")
        ax.set_ylabel("objective J")
        ax.set_title("stairs-medium: counterfactual, NOT measured", fontsize=8)
        ax.legend(fontsize=7)
        _save(fig, out)


def outcome_bars(outc: pd.DataFrame, setting: str, out: str | Path) -> None:
    d = outc[outc["setting"] == setting].set_index("batch")
    cols = ["kept", "rolled_back", "restored", "noop"]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        x = np.arange(len(d))
        wdt = 0.2
        for i, c in enumerate(cols):
            ax.bar(x + (i - 1.5) * wdt, d[c], wdt, label=c)
        ax.set_xticks(x)
        ax.set_xticklabels(d.index)
        ax.set_ylabel("reports per run")
        ax.set_title(setting, fontsize=8)
        ax.legend(fontsize=7)
        _save(fig, out)


def load_batches(batches: dict[str, tuple[str | Path, list[str] | None]]) -> pd.DataFrame:
    """Run table with a ``batch`` column; ``batches`` maps label -> (root, run_ids|None)."""
    parts = []
    for label, (root, ids) in batches.items():
        t = load_coach_table(root, ids)
        t["batch"] = label
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=float))
