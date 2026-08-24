#!/usr/bin/env python
"""Evaluate a trained checkpoint on its (or another) terrain.

Example:
    python scripts/evaluate.py --checkpoint data/results/<run_id>/checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path

import yaml

from quadruped_rl.harness.evaluator import Evaluator
from quadruped_rl.harness.seeding import set_global_seed
from quadruped_rl.registry import get_algorithm, get_env_backend


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--terrain-level", default=None, choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=1000)  # eval seeds disjoint from training
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if args.episodes:
        cfg["run"]["eval_episodes"] = args.episodes

    set_global_seed(args.seed)
    env = get_env_backend(cfg["sim"]["backend"])(cfg)
    algo = get_algorithm(cfg["algorithm"]["name"])(cfg, env.observation_dim, env.action_dim)
    algo.load(ckpt)

    metrics = Evaluator(cfg, env).run(algo)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
