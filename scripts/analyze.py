#!/usr/bin/env python
"""Statistical analysis + figures for a completed experiment.

Example:
    python scripts/analyze.py --experiment phase2_matrix \
        --metrics success_rate cost_of_transport mean_forward_velocity_ms
"""

import argparse
from pathlib import Path

from quadruped_rl.analysis.plots import comparison_box, metric_heatmap
from quadruped_rl.analysis.statistics import compare_algorithms, load_results_table


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--results-root", default="data/results")
    p.add_argument("--metrics", nargs="+", default=["success_rate"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    df = load_results_table(args.results_root, args.experiment)
    if df.empty:
        raise SystemExit(f"No completed runs found for '{args.experiment}'")
    out_dir = Path(args.out or f"data/results/analysis_{args.experiment}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results_table.csv", index=False)

    for metric in args.metrics:
        if metric not in df.columns:
            print(f"[skip] metric '{metric}' not in results")
            continue
        report = compare_algorithms(df, metric)
        print(f"\n=== {metric} ===")
        print("ANOVA:", report["anova"])
        print(report["tukey"].to_string(index=False))
        for name, s in report["summary"].items():
            lo, hi = s["ci95"]
            print(
                f"  {name}: {s['mean']:.4f} ± {s['std']:.4f} "
                f"(95% CI [{lo:.4f}, {hi:.4f}], n={s['n']})"
            )
        report["tukey"].to_csv(out_dir / f"tukey_{metric}.csv", index=False)
        comparison_box(df, metric, out_dir / f"box_{metric}")
        metric_heatmap(df, metric, out_dir / f"heatmap_{metric}")
    print(f"\nSaved analysis to {out_dir}")


if __name__ == "__main__":
    main()
