#!/usr/bin/env python
"""Generate the reward-coach study job list (Phase 3 pilot).

Two settings, four conditions each (fixed | random | hillclimb | llm), 3 seeds:
  A "recovery": naive reward on flat        -> can the coach repair a broken reward?
  B "barrier" : traditional reward on rough-hard and stairs-medium
                (0 % success for every algorithm in Phase 1) -> can it break the wall?

The python interpreter is taken from $ISAAC_PY at run time so the same job
file runs locally (conda env_isaaclab) and on the 4x4090 server (Isaac Sim
4.5 bundled python). Consume with scripts/run_jobs.py --gpus 0,1,2,3.

    python scripts/make_coach_jobs.py --out data/results/coach_batch/jobs.txt
    # extend an existing batch (appended lines keep earlier job indices stable):
    python scripts/make_coach_jobs.py --out ... --append --settings B --terrains rough \
        --seeds 5 6 7 8 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

CONDITIONS = ["none", "random", "hillclimb", "llm"]
SETTINGS = [
    # (tag, terrain, level, reward)
    ("A", "flat", "easy", "naive"),
    ("B", "rough", "hard", "traditional"),
    ("B", "stairs", "medium", "traditional"),
]
PY = "${ISAAC_PY:-$HOME/anaconda3/envs/env_isaaclab/bin/python}"


def job(terrain: str, level: str, reward: str, coach: str, seed: int, steps: int) -> str:
    parts = [
        f"PYTHONPATH=src {PY} scripts/train.py --sim isaaclab --algorithm ppo --robot a1",
        f"--terrain {terrain} --reward {reward} --seed {seed}",
        f"--override sim.terrain_level={level}",
        f"--override run.total_timesteps={steps}",
        "--override run.eval_interval_steps=1000000",
        "--override run.checkpoint_interval_steps=10000000",
        "--override logging.wandb=false",
    ]
    if coach != "none":
        parts.append(f"--coach {coach}")
    return " ".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=40_000_000)
    p.add_argument("--settings", nargs="+", default=["A", "B"])
    p.add_argument("--conditions", nargs="+", default=CONDITIONS)
    p.add_argument("--terrains", nargs="+", default=None, help="restrict to these terrains")
    p.add_argument("--append", action="store_true", help="append to --out (extend a batch)")
    args = p.parse_args()

    lines = []
    for tag, terrain, level, reward in SETTINGS:
        if tag not in args.settings or (args.terrains and terrain not in args.terrains):
            continue
        for coach in args.conditions:
            for seed in args.seeds:
                lines.append(job(terrain, level, reward, coach, seed, args.steps))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a" if args.append else "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {out}{' (appended)' if args.append else ''}")


if __name__ == "__main__":
    main()
