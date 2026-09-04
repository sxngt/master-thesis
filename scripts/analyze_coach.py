#!/usr/bin/env python
"""Statistics + figures for the reward-coach study (Phase 3 pilot).

Groups runs by setting (terrain-level-reward) and coach condition
(none | random | hillclimb | llm), then per setting: ANOVA -> Tukey HSD ->
Cohen's d on the objective J and success rate, objective learning curves,
parameter trajectories of every coached run, and a flat intervention table.

    uv run python scripts/analyze_coach.py --results-root data/results/coach_batch/runs
    uv run python scripts/analyze_coach.py --jobs data/results/coach_batch/jobs.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

from quadruped_rl.analysis.coach import (
    compare_conditions,
    intervention_table,
    load_coach_table,
    objective_curves,
    parameter_trajectories,
)


def _coach_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())["coach"]


def _bounds(coach_cfg: dict) -> dict[str, tuple[float, float]]:
    return {k: (float(v["low"]), float(v["high"])) for k, v in coach_cfg["params"].items()}


def _run_ids_from_jobs(jobs: str) -> list[str]:
    """run_jobs.py status -> run ids, read from each job's log (train.py's final JSON)."""
    status = json.loads(Path(jobs + ".status.json").read_text())
    ids = []
    for v in status.values():
        log = Path(v.get("log", "")) if isinstance(v, dict) else None
        if log and log.exists():
            # last match: a log may carry output of an earlier, superseded process
            found = re.findall(r'"run_id":\s*"([^"]+)"', log.read_text(errors="ignore"))
            if found:
                ids.append(found[-1])
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", default="data/results")
    p.add_argument("--jobs", default=None, help="restrict to runs listed in <jobs>.status.json")
    p.add_argument("--metrics", nargs="+", default=["objective_final", "success_rate"])
    p.add_argument("--coach-config", default="configs/coach/llm.yaml")
    p.add_argument("--out", default="data/results/analysis_coach")
    args = p.parse_args()

    coach_cfg = _coach_cfg(args.coach_config)
    run_ids = _run_ids_from_jobs(args.jobs) if args.jobs else None
    table = load_coach_table(args.results_root, run_ids, weights=coach_cfg["objective"])
    if table.empty:
        raise SystemExit("no completed coach runs found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    public = [c for c in table.columns if not c.startswith("_")]
    table[public].to_csv(out / "results_table.csv", index=False)
    print(
        table.groupby(["setting", "condition"])[["objective_final", "success_rate", "n_kept"]]
        .agg(["mean", "std", "count"])
        .to_string()
    )

    for metric in args.metrics:
        for setting, rep in compare_conditions(table, metric).items():
            print(f"\n=== {setting} / {metric} ===")
            print("ANOVA:", rep["anova"])
            print(rep["tukey"].to_string(index=False))
            for name, s in rep["summary"].items():
                lo, hi = s["ci95"]
                ci = f"95% CI [{lo:.4f}, {hi:.4f}]"
                print(f"  {name}: {s['mean']:.4f} ± {s['std']:.4f} ({ci}, n={s['n']})")
            rep["tukey"].to_csv(out / f"tukey_{setting}_{metric}.csv", index=False)

    for setting in sorted(table["setting"].unique()):
        objective_curves(table, setting, out / f"objective_{setting}")
    bounds = _bounds(coach_cfg)
    for _, r in table[table["condition"] != "none"].iterrows():
        parameter_trajectories(r, bounds, out / "params" / r["run_id"])
    its = intervention_table(table)
    its.to_csv(out / "interventions.csv", index=False)
    print(f"\n{len(its)} interventions; saved analysis to {out}")


if __name__ == "__main__":
    main()
