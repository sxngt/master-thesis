"""Training harness: config -> env -> algorithm -> logging/checkpoints.

Single entry point for every training run (scripts/train.py is a thin CLI).
Guarantees: global seeding, resolved-config persistence, periodic evaluation,
checkpointing, and metric logging — identical across all algorithms.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from quadruped_rl.harness.checkpoints import CheckpointManager
from quadruped_rl.harness.config import save_resolved_config
from quadruped_rl.harness.evaluator import Evaluator
from quadruped_rl.harness.logging_utils import RunLogger
from quadruped_rl.harness.seeding import set_global_seed
from quadruped_rl.registry import get_algorithm, get_env_backend

DATA_ROOT = Path(__file__).resolve().parents[3] / "data"


def make_run_id(cfg: dict[str, Any]) -> str:
    algo = cfg["algorithm"]["name"]
    robot = cfg["robot"]["name"]
    terrain = cfg["terrain"]["name"]
    seed = cfg["run"]["seed"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{algo}_{robot}_{terrain}_s{seed}_{stamp}_{uuid.uuid4().hex[:6]}"


class Trainer:
    def __init__(self, cfg: dict[str, Any], run_dir: str | Path | None = None):
        self.cfg = cfg
        if cfg["run"].get("smoke_test"):
            self._apply_smoke_overrides()

        self.run_id = make_run_id(cfg)
        self.run_dir = Path(run_dir) if run_dir else DATA_ROOT / "results" / self.run_id
        save_resolved_config(cfg, self.run_dir)  # reproducibility: never skip

        set_global_seed(cfg["run"]["seed"])

        env_cls = get_env_backend(cfg["sim"]["backend"])
        self.env = env_cls(cfg)
        algo_cls = get_algorithm(cfg["algorithm"]["name"])
        self.algorithm = algo_cls(cfg, self.env.observation_dim, self.env.action_dim)

        self.logger = RunLogger(self.run_dir, cfg, self.run_id)
        self.checkpoints = CheckpointManager(self.run_dir)
        self.evaluator = Evaluator(cfg, self.env)

    def _apply_smoke_overrides(self) -> None:
        """~1-minute CPU-safe sanity run. Never used for reported results."""
        run, sim = self.cfg["run"], self.cfg["sim"]
        run.update(
            total_timesteps=2_000,
            eval_interval_steps=1_000,
            checkpoint_interval_steps=1_000,
            eval_episodes=2,
            device="cpu",
        )
        sim.update(backend=sim.get("smoke_backend", "mock"), num_envs=4)

    def train(self) -> dict[str, Any]:
        cfg_run = self.cfg["run"]
        total = cfg_run["total_timesteps"]
        step = 0
        next_eval = cfg_run["eval_interval_steps"]
        next_ckpt = cfg_run["checkpoint_interval_steps"]
        last_eval: dict[str, Any] = {}

        obs = self.env.reset()
        while step < total:
            obs, train_metrics, collected = self.algorithm.collect_and_update(self.env, obs)
            step += collected
            self.logger.log({f"train/{k}": v for k, v in train_metrics.items()}, step)

            if step >= next_eval:
                last_eval = self.evaluator.run(self.algorithm)
                self.logger.log({f"eval/{k}": v for k, v in last_eval.items()}, step)
                next_eval += cfg_run["eval_interval_steps"]

            if step >= next_ckpt:
                self.checkpoints.save(self.algorithm, step, last_eval)
                next_ckpt += cfg_run["checkpoint_interval_steps"]

        final = self.evaluator.run(self.algorithm)
        self.checkpoints.save(self.algorithm, step, final)
        self.logger.log({f"final/{k}": v for k, v in final.items()}, step)
        self.logger.close()
        return {"run_id": self.run_id, "run_dir": str(self.run_dir), "final": final}
