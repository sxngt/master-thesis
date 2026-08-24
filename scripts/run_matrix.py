#!/usr/bin/env python
"""Run (or resume) a full experiment matrix, or export per-cell job commands.

Examples:
    python scripts/run_matrix.py --experiment phase2_matrix
    python scripts/run_matrix.py --experiment phase2_matrix --export-jobs jobs.txt
"""

import argparse

from quadruped_rl.harness.matrix_runner import MatrixRunner


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument(
        "--export-jobs",
        default=None,
        metavar="PATH",
        help="write one train command per cell for cluster dispatch",
    )
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()

    overrides = {"run": {"smoke_test": True}} if args.smoke_test else {}
    runner = MatrixRunner(args.experiment, overrides=overrides)
    if args.export_jobs:
        out = runner.export_jobs(args.export_jobs)
        print(f"wrote {out}")
    else:
        runner.run(resume=not args.no_resume)


if __name__ == "__main__":
    main()
