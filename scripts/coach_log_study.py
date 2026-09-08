#!/usr/bin/env python
"""Log study of the LLM reward coach over every finished batch — no new training.

Collects every coach report (``coach_log.jsonl``) and evaluation trace of the
batches below, then writes measured tables/figures (learning curves, J
decomposition, intervention outcomes, parameter usage, token cost) and one
clearly labelled *estimate*: what the stairs LLM runs would score if the
coach restored the commanded speed it lowered (coach v4.1 behaviour), using
the command-tracking ratio fitted on the logs.

    uv run python scripts/coach_log_study.py --out data/results/coach_llm_study
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from analyze_coach import _run_ids_from_jobs

from quadruped_rl.analysis.coach import CONDITION_ORDER, compare_conditions
from quadruped_rl.analysis.coach_study import (
    TARGET_KEY,
    action_table,
    collect_calls,
    cost_table,
    counterfactual_objective,
    counterfactual_plot,
    decomposition,
    decomposition_bars,
    load_batches,
    objective_curves_by_batch,
    outcome_bars,
    outcome_table,
    target_timelines,
    tracking_fit,
    tracking_scatter,
    write_json,
)
from quadruped_rl.analysis.statistics import confidence_interval

BATCHES = {  # label -> (runs root, jobs file)
    "v1": ("data/results/coach_batch/runs", "data/results/coach_batch/jobs.txt"),
    "v2": ("data/results/coach_v2/runs", "data/results/coach_v2/jobs.txt"),
    "v3": ("data/results/coach_v3/runs", "data/results/coach_v3/jobs.txt"),
    "v4": ("data/results/coach_v4/runs", "data/results/coach_v4/jobs.txt"),
}
STAIRS = "stairs-medium-traditional"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/results/coach_llm_study")
    p.add_argument("--usd-per-m-in", type=float, default=2.5, help="gpt-5.4 list price")
    p.add_argument("--usd-per-m-out", type=float, default=15.0)
    p.add_argument("--krw-per-usd", type=float, default=1390.0)
    args = p.parse_args()
    out = Path(args.out)
    figs = out / "figs"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- calls
    calls = collect_calls({b: root for b, (root, _) in BATCHES.items()})
    calls.drop(columns=["applied"]).to_csv(out / "calls.csv", index=False)
    actions = action_table(calls)
    actions.to_csv(out / "actions.csv", index=False)
    cost = cost_table(calls, args.usd_per_m_in, args.usd_per_m_out, args.krw_per_usd)
    cost.to_csv(out / "cost.csv", index=False)
    outc = outcome_table(calls)
    outc.to_csv(out / "outcomes.csv", index=False)
    print("== token cost per batch\n", cost.round(2).to_string(index=False))
    print("== report outcomes per run\n", outc.to_string(index=False))

    # ------------------------------------------------- runs (measured, v3/v4)
    table = load_batches(
        {b: (BATCHES[b][0], _run_ids_from_jobs(BATCHES[b][1])) for b in ["v3", "v4"]}
    )
    table = table[~((table["batch"] == "v4") & (table["condition"] == "none"))]
    keep = [c for c in table.columns if not c.startswith("_")]
    table[keep].to_csv(out / "results_table.csv", index=False)
    dec = decomposition(table.assign(condition=table["condition"] + "_" + table["batch"]))
    dec.to_csv(out / "decomposition.csv", index=False)
    print("== J decomposition (final)\n", dec.round(3).to_string(index=False))

    stats = {}
    for setting, d in table.groupby("setting"):
        # v4 coach conditions vs the v3 controls: the comparison the thesis reports
        cmp = d[(d["batch"] == "v4") | (d["condition"] == "none")]
        if cmp["condition"].nunique() >= 2 and (cmp.groupby("condition").size() >= 2).all():
            res = compare_conditions(cmp, "objective_final")[setting]
            stats[setting] = {
                "anova": res["anova"],
                "summary": res["summary"],
                "tukey": res["tukey"].to_dict("records")
                if hasattr(res["tukey"], "to_dict")
                else res["tukey"],
            }
    write_json(stats, out / "stats_v4_vs_none.json")

    groups = [
        ("none (v3)", "none", "v3"),
        ("LLM v3", "llm", "v3"),
        ("LLM v4", "llm", "v4"),
        ("random (v4)", "random", "v4"),
        ("hillclimb (v4)", "hillclimb", "v4"),
    ]
    for setting in sorted(table["setting"].unique()):
        objective_curves_by_batch(table, setting, figs / f"curves_{setting}", groups)
        target_timelines(calls, setting, figs / f"target_{setting}")
        outcome_bars(outc, setting, figs / f"outcomes_{setting}")
    decomposition_bars(
        dec, figs / "decomposition", [f"{c}_{b}" for c in CONDITION_ORDER for b in ["v3", "v4"]]
    )

    # -------------------------------- parameter usage by setting (LLM, v3+v4)
    use = (
        actions[actions["batch"].isin(["v3", "v4"]) & (actions["status"] == "kept")]
        .groupby(["setting", "param"])
        .size()
        .unstack(fill_value=0)
    )
    use.to_csv(out / "param_usage_kept.csv")
    print("== kept parameter changes by setting (v3+v4)\n", use.to_string())
    dj = (
        actions[actions["batch"].isin(["v3", "v4"])]
        .dropna(subset=["delta_j"])
        .groupby(["setting", "direction"])["delta_j"]
        .agg(["size", "mean", "std"])
    )
    dj.to_csv(out / "delta_j_by_direction.csv")
    print("== delta J after an intervention, by target_ms direction\n", dj.round(3).to_string())

    # ------------------------------------------- estimate: stairs with v4.1
    fit = tracking_fit(calls[calls["batch"].isin(["v3", "v4"])])
    write_json(fit, out / "tracking_fit.json")
    tracking_scatter(calls[calls["batch"].isin(["v3", "v4"])], fit, figs / "tracking")
    print("== command tracking fit (success >= 0.5):", {k: round(v, 3) for k, v in fit.items()})
    st = table[(table["setting"] == STAIRS)]
    llm4 = st[(st["condition"] == "llm") & (st["batch"] == "v4")]
    none = st[st["condition"] == "none"]
    targets = np.round(np.arange(0.6, 1.51, 0.05), 2)
    cf = counterfactual_objective(
        llm4["success_rate"].to_numpy(), targets, fit["slope_through_origin"]
    )
    cf["assumption"] = "measured success kept; velocity = ratio x target"
    cf.to_csv(out / "counterfactual_stairs.csv", index=False)
    none_mean, lo, hi = confidence_interval(none["objective_final"].to_numpy())
    counterfactual_plot(cf, none_mean, (lo, hi), figs / "counterfactual_stairs")
    print("== ESTIMATE stairs LLM v4 seeds re-scored at restored target (not measured)")
    print(cf[cf["target_ms"].isin([0.7, 1.0, 1.2, 1.5])].round(3).to_string(index=False))
    print(f"   none measured: {none_mean:.3f} [{lo:.3f}, {hi:.3f}] (n={len(none)})")
    print(
        f"   {TARGET_KEY} final values, stairs LLM v4:",
        sorted(llm4["_final_params"].map(lambda p: (p or {}).get(TARGET_KEY)).tolist()),
    )
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
