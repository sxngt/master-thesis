#!/usr/bin/env python
"""Cross-simulator validation: evaluate one trained checkpoint across
multiple simulation backends and report the sim-to-sim gap per metric.

Core to the research design — policies are trained in Isaac Gym and
verified in PyBullet/Gazebo with identified physics parameters.

Example:
    python scripts/cross_validate.py \
        --checkpoint data/results/<run_id>/checkpoints/best.pt \
        --sims isaacgym pybullet gazebo
"""

import argparse
import json
from pathlib import Path

import yaml

from quadruped_rl.harness.config import CONFIG_ROOT, deep_merge, load_yaml
from quadruped_rl.harness.evaluator import Evaluator
from quadruped_rl.harness.seeding import set_global_seed
from quadruped_rl.registry import get_algorithm, get_env_backend


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--sims", nargs="+", required=True, help="backend config names from configs/sim/"
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    base_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    base_cfg["run"]["eval_episodes"] = args.episodes

    results: dict[str, dict] = {}
    for sim in args.sims:
        cfg = deep_merge(base_cfg, load_yaml(CONFIG_ROOT / "sim" / f"{sim}.yaml"))
        set_global_seed(args.seed)
        try:
            env = get_env_backend(cfg["sim"]["backend"])(cfg)
        except (KeyError, NotImplementedError, ImportError) as e:
            print(f"[skip] {sim}: {e}")
            continue
        algo = get_algorithm(cfg["algorithm"]["name"])(cfg, env.observation_dim, env.action_dim)
        algo.load(ckpt)
        results[sim] = Evaluator(cfg, env).run(algo)
        print(f"[{sim}] {json.dumps(results[sim], indent=2)}")

    if len(results) >= 2:
        sims = list(results)
        ref = sims[0]
        print(f"\n=== sim-to-sim gap (relative to {ref}) ===")
        for other in sims[1:]:
            for metric, ref_val in results[ref].items():
                if metric in results[other] and ref_val:
                    gap = (results[other][metric] - ref_val) / abs(ref_val)
                    print(f"  {other}/{metric}: {gap:+.1%}")

    out = Path(args.out or run_dir / "cross_validation.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
