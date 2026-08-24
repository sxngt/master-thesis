#!/usr/bin/env python
"""Optuna hyperparameter sweep for one algorithm/robot/terrain cell.

Example:
    python scripts/sweep.py --algorithm sac --robot a1 --terrain stairs --trials 50
"""

import argparse

from quadruped_rl.harness.sweep import run_sweep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--terrain", required=True)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--metric", default="success_rate")
    p.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="shortened budget per trial (default: full config value)",
    )
    p.add_argument("--storage", default=None, help="Optuna storage URL for distributed sweeps")
    args = p.parse_args()

    study = run_sweep(
        args.algorithm,
        args.robot,
        args.terrain,
        n_trials=args.trials,
        objective_metric=args.metric,
        timesteps=args.timesteps,
        storage=args.storage,
    )
    print("best value:", study.best_value)
    print("best params:", study.best_params)


if __name__ == "__main__":
    main()
