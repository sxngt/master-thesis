#!/usr/bin/env python
"""Record a rollout video (MP4 + optional GIF) of a trained checkpoint.

Runs the policy deterministically in the Isaac Lab backend with an
offscreen chase camera following the robot. Requires the Isaac Lab Python.

Example:
    PYTHONPATH=src ~/anaconda3/envs/env_isaaclab/bin/python scripts/record_video.py \
        --checkpoint data/results/<run>/checkpoints/step_....pt \
        --out docs/media/ppo_flat.mp4 --seconds 10 --gif
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output .mp4 path")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--gif", action="store_true", help="also write a compressed GIF next to the MP4")
    args = p.parse_args()

    import imageio

    from quadruped_rl.harness.seeding import set_global_seed
    from quadruped_rl.registry import get_algorithm, get_env_backend

    ckpt = Path(args.checkpoint)
    run_dir = ckpt.parent.parent if ckpt.parent.name == "checkpoints" else ckpt.parent
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    cfg["sim"].update(render=True, num_envs=1, headless=True)
    cfg["logging"]["wandb"] = False

    set_global_seed(args.seed)
    env = get_env_backend("isaaclab")(cfg)
    algo = get_algorithm(cfg["algorithm"]["name"])(cfg, env.observation_dim, env.action_dim)
    algo.load(ckpt)

    fps = int(round(1.0 / env.control_dt))  # control rate (50 Hz)
    steps = int(args.seconds / env.control_dt)

    # Warm-up: RTX shaders compile asynchronously — frames are black
    # placeholders until compilation finishes. Step until a real frame lands.
    obs = env.reset()
    import time

    deadline = time.time() + 600
    while time.time() < deadline:
        obs, _, _, _ = env.step(algo.act(obs, deterministic=True))
        frame = env.render()
        if frame is not None and np.asarray(frame).max() > 0:
            print("renderer warm", flush=True)
            break
    else:
        raise RuntimeError("renderer produced only black frames for 600 s")

    obs = env.reset()
    frames = []
    for _ in range(steps):
        actions = algo.act(obs, deterministic=True)
        obs, _, _, _ = env.step(actions)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame)[..., :3])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out, frames, fps=fps, codec="libx264", quality=7)
    meta = {
        "checkpoint": str(ckpt),
        "algorithm": cfg["algorithm"]["name"],
        "terrain": cfg["terrain"]["name"],
        "seconds": args.seconds,
        "frames": len(frames),
    }
    print(json.dumps(meta), flush=True)

    if args.gif:
        # compact inline-friendly GIF: 10 fps, 384 px wide, first 6 s
        gif_frames = []
        stride = max(1, fps // 10)
        for f in frames[: int(6.0 * fps) : stride]:
            h, w = f.shape[:2]
            scale = 384 / w
            small = f[:: max(1, int(1 / scale)), :: max(1, int(1 / scale))]
            gif_frames.append(small)
        imageio.mimwrite(out.with_suffix(".gif"), gif_frames, fps=10, loop=0)
        print(f"gif: {out.with_suffix('.gif')}", flush=True)

    # Isaac Sim shutdown deadlocks with cameras enabled (plugin unload hang)
    # — outputs are already on disk, exit hard instead of env.close().
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
