"""Analysis of reward-coach runs (Phase 3 pilot).

A coach run is grouped by *setting* (terrain, level, reward) and *condition*
(coach strategy: none | random | hillclimb | llm). Everything here is pure
pandas/numpy over ``data/results/<run>/{config.yaml,metrics.jsonl,coach_log.jsonl}``
so it can be unit-tested without a simulator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from quadruped_rl.analysis.plots import STYLE, _save
from quadruped_rl.analysis.statistics import compare_algorithms

# objective shared by every condition (configs/coach/*.yaml); the fixed
# condition has no coach block, so it is recomputed here from eval metrics
DEFAULT_OBJECTIVE = {"success_rate": 1.0, "mean_forward_velocity_ms": 0.5}
CONDITION_ORDER = ["none", "random", "hillclimb", "llm"]


def objective_from_eval(rec: dict[str, float], weights: dict[str, float], prefix: str) -> float:
    return float(sum(w * rec.get(f"{prefix}{k}", 0.0) for k, w in weights.items()))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(run_dir: Path, weights: dict[str, float] | None = None) -> dict[str, Any] | None:
    """One row per run: identity, final metrics, objective trace, coach summary.

    ``weights`` defines J for *every* run (one objective across conditions);
    defaults to DEFAULT_OBJECTIVE, not the run's own coach config.
    """
    cfg_path, metrics_path = run_dir / "config.yaml", run_dir / "metrics.jsonl"
    if not (cfg_path.exists() and metrics_path.exists()):
        return None
    cfg = yaml.safe_load(cfg_path.read_text())
    coach_cfg = cfg.get("coach") or {}
    weights = weights or DEFAULT_OBJECTIVE
    recs = _read_jsonl(metrics_path)
    evals = [r for r in recs if "eval/success_rate" in r]
    finals = [r for r in recs if "final/success_rate" in r]
    if not finals:
        return None
    final = finals[-1]
    steps = np.array([r["step"] for r in evals], dtype=float)
    obj = np.array([objective_from_eval(r, weights, "eval/") for r in evals])
    log = _read_jsonl(run_dir / "coach_log.jsonl")
    # coach_log has one "pending" line at apply time and one settled line
    # (kept | rolled_back) at the next eval — keep the settled state per k
    settled: dict[int, dict[str, Any]] = {}
    for rec in log:
        if rec["status"] != "pending" or rec["k"] not in settled:
            settled[rec["k"]] = rec
    interventions = [r for r in settled.values() if r["status"] != "noop"]
    row = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "algorithm": cfg["algorithm"]["name"],
        "terrain": cfg["terrain"]["name"],
        "level": cfg["sim"].get("terrain_level", "easy"),
        "reward": cfg.get("reward", {}).get("name", "traditional"),
        "condition": coach_cfg.get("strategy", "none"),
        "seed": cfg["run"]["seed"],
        "objective_final": objective_from_eval(final, weights, "final/"),
        "objective_best": float(obj.max()) if len(obj) else float("nan"),
        "objective_auc": float(np.trapezoid(obj, steps) / max(steps[-1], 1.0))
        if len(obj) > 1
        else 0.0,
        "n_interventions": len(interventions),
        "n_kept": sum(r["status"] == "kept" for r in interventions),
        "n_rolled_back": sum(r["status"] == "rolled_back" for r in interventions),
        "n_pending": sum(r["status"] == "pending" for r in interventions),  # unsettled at end
        "tokens_in": sum((r.get("usage") or {}).get("input_tokens", 0) for r in log),
        "tokens_out": sum((r.get("usage") or {}).get("output_tokens", 0) for r in log),
        "_steps": steps,
        "_objective": obj,
        "_interventions": interventions,
        "_final_params": (log[-1].get("params_before") if log else None),
    }
    for k, v in final.items():
        if k.startswith("final/"):
            row[k.removeprefix("final/")] = v
    row["setting"] = f"{row['terrain']}-{row['level']}-{row['reward']}"
    return row


def load_coach_table(
    results_root: str | Path,
    run_ids: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    for d in sorted(Path(results_root).iterdir()):
        if run_ids is not None and d.name not in run_ids:
            continue
        r = load_run(d, weights)
        if r is not None:
            rows.append(r)
    return pd.DataFrame(rows)


def compare_conditions(table: pd.DataFrame, metric: str) -> dict[str, dict[str, Any]]:
    """Per setting: ANOVA + Tukey HSD + Cohen's d across coach conditions."""
    out = {}
    for setting, g in table.groupby("setting"):
        if g["condition"].nunique() < 2 or (g.groupby("condition").size() < 2).any():
            continue
        out[setting] = compare_algorithms(g, metric, group_col="condition")
    return out


def intervention_table(table: pd.DataFrame) -> pd.DataFrame:
    """Flat list of every settled intervention (for the thesis appendix)."""
    rows = []
    for _, r in table.iterrows():
        for it in r["_interventions"]:
            rows.append(
                {
                    "setting": r["setting"],
                    "condition": r["condition"],
                    "seed": r["seed"],
                    "k": it["k"],
                    "step": it["step"],
                    "status": it["status"],
                    "n_applied": len(it["applied"]),
                    "applied": "; ".join(f"{k}={v:.3g}" for k, v in it["applied"].items()),
                    "objective_before": it.get("objective_before"),
                    "objective_after": it.get("objective_after"),
                    "confidence": it.get("confidence"),
                    "diagnosis": (it.get("diagnosis") or "")[:200],
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figures
def objective_curves(table: pd.DataFrame, setting: str, out: str | Path) -> None:
    """Mean ± std objective over training, one line per condition."""
    g = table[table["setting"] == setting]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for cond in CONDITION_ORDER:
            runs = g[g["condition"] == cond]
            runs = runs[runs["_steps"].map(len) > 1]
            if runs.empty:
                continue
            grid = runs.iloc[0]["_steps"]
            stacked = np.stack(
                [np.interp(grid, r["_steps"], r["_objective"]) for _, r in runs.iterrows()]
            )
            mean, std = stacked.mean(0), stacked.std(0)
            ax.plot(grid, mean, label=f"{cond} (n={len(runs)})")
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Objective J")
        ax.set_title(setting)
        ax.legend(fontsize=7)
        _save(fig, out)


def parameter_trajectories(
    run_row: pd.Series, bounds: dict[str, tuple[float, float]], out: str | Path
) -> None:
    """Kept parameter values over training for one run, normalised to [0, 1]
    within the coach bounds so every parameter shares one axis."""
    its = [it for it in run_row["_interventions"] if it["status"] == "kept"]
    if not its:
        return
    keys = sorted({k for it in its for k in it["applied"]})
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for key in keys:
            lo, hi = bounds.get(key, (0.0, 1.0))
            xs, ys = [0.0], [(its[0]["params_before"][key] - lo) / (hi - lo)]
            for it in its:
                if key in it["applied"]:
                    xs.append(it["step"])
                    ys.append((it["applied"][key] - lo) / (hi - lo))
            ax.step(xs, ys, where="post", label=key)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Normalised parameter")
        ax.set_title(f"{run_row['setting']} / {run_row['condition']} / seed {run_row['seed']}")
        ax.legend(fontsize=6, ncol=2)
        _save(fig, out)
