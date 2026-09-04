#!/usr/bin/env python
"""Train one (algorithm, robot, terrain, seed) combination.

Examples:
    python scripts/train.py --algorithm ppo --robot a1 --terrain stairs --seed 0
    python scripts/train.py --algorithm ppo --robot a1 --terrain flat --smoke-test
"""

import argparse
import json

from quadruped_rl.harness.config import compose_config
from quadruped_rl.harness.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sim",
        default=None,
        help="simulation backend config (configs/sim/): isaacgym|pybullet|gazebo|mock",
    )
    p.add_argument("--algorithm", required=True)
    p.add_argument("--robot", required=True)
    p.add_argument("--terrain", required=True)
    p.add_argument("--reward", default="traditional")
    p.add_argument(
        "--coach",
        default=None,
        help="reward scheduler config (configs/coach/): llm|random|hillclimb (default: none)",
    )
    p.add_argument("--experiment", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None, help="unused convenience alias for --experiment path")
    p.add_argument(
        "--smoke-test", action="store_true", help="~1-minute CPU sanity run on the mock env"
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dot-path config override, e.g. run.total_timesteps=1000",
    )
    args = p.parse_args()

    overrides: dict = {"run": {"seed": args.seed, "smoke_test": args.smoke_test}}
    for item in args.override:
        key, _, value = item.partition("=")
        node = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            node[parts[-1]] = json.loads(value)
        except json.JSONDecodeError:
            node[parts[-1]] = value

    cfg = compose_config(
        sim=args.sim,
        algorithm=args.algorithm,
        robot=args.robot,
        terrain=args.terrain,
        reward=args.reward,
        coach=args.coach,
        experiment=args.experiment,
        overrides=overrides,
    )
    result = Trainer(cfg).train()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
