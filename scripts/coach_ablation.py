#!/usr/bin/env python
"""Ablation batch: LLM-free coaches (random / hillclimb) on the *same* decision
layer as an LLM coach version, same settings x seeds as the evolution loop.

Answers "is the gain the LLM's, or would any bounded reward tuner do?": the
control arms share cadence, bounds, objective and every decision-layer
override of the version (``coach.version_dir`` applies its coach.yaml to any
strategy; the prompt templates and the playbook text are simply unused).
Jobs are written in the evolution driver's format and run with
scripts/run_jobs.py, so results land in ``<root>/<arm>/runs/<run>/`` exactly
like ``data/results/evolve*/<version>/runs`` and can be paired with them by
(setting, seed) through analysis.coach / evolve.paired_compare.

On dongbeen, after the evolution driver has finished with the training GPU:
    export ISAAC_PY=/mnt/sdb1/sxngt/isaac-sim-4.5.0/python.sh
    nohup $ISAAC_PY scripts/coach_ablation.py --root data/results/ablation_v7p \\
        --version v7p --arms random hillclimb --sim-gpus 3 --parallel 3 \\
        >> data/results/ablation_v7p/driver.log 2>&1 &
``--dry-run`` only writes the jobs files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import evolve_coach as ev  # noqa: E402

ARMS = ("random", "hillclimb", "none")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", default="data/results/ablation")
    p.add_argument("--version", default="v7p", help="version whose coach.yaml the arms inherit")
    p.add_argument("--versions-dir", default=ev.DEFAULT_VERSIONS)
    p.add_argument("--arms", nargs="+", default=["random", "hillclimb"], choices=ARMS)
    p.add_argument("--python", default=ev.DEFAULT_PY)
    p.add_argument("--settings", nargs="+", default=ev.DEFAULT_SETTINGS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=40_000_000)
    p.add_argument("--eval-interval", type=int, default=1_000_000)
    p.add_argument("--extra-override", action="append", default=[])
    p.add_argument("--sim", default="isaaclab")
    p.add_argument("--sim-gpus", default="3")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="write the jobs files only")
    a = p.parse_args()

    # job_line() reads these from the driver's namespace; no LLM URL/model for
    # LLM-free arms (coach != args.coach -> the llm block is left alone)
    root = (REPO / a.root).resolve()  # job_line() wants an absolute root under the repo
    args = SimpleNamespace(
        python=a.python,
        sim=a.sim,
        steps=a.steps,
        eval_interval=a.eval_interval,
        root=str(root),
        versions_dir=a.versions_dir,
        coach="llm_local",
        llm_model=None,
        llm_url=[],
        extra_override=a.extra_override,
        sim_gpus=a.sim_gpus,
        parallel=a.parallel,
    )
    root.mkdir(parents=True, exist_ok=True)
    for arm in a.arms:
        runs = root / arm / "runs"
        version = None if arm == "none" else a.version
        lines = [
            ev.job_line(args, s, seed, arm, version, runs, i)
            for i, (s, seed) in enumerate((s, seed) for s in a.settings for seed in a.seeds)
        ]
        jobs = root / arm / "jobs.txt"
        jobs.parent.mkdir(parents=True, exist_ok=True)
        if not jobs.exists():
            jobs.write_text("\n".join(lines) + "\n")
        print(f"{arm}: {len(lines)} jobs -> {jobs.relative_to(REPO)}")
        if not a.dry_run:
            ev.run_jobs(args, jobs, lines)
            rows = ev.rows_of(runs)
            print(f"{arm}: {len(rows)}/{len(lines)} runs finished")


if __name__ == "__main__":
    main()
