"""Experiment logging: W&B when available/enabled, always local JSONL fallback."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str | Path, cfg: dict[str, Any], run_name: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.run_dir / "metrics.jsonl", "a")
        self._wandb = None
        if cfg.get("logging", {}).get("wandb", False):
            try:
                import wandb

                self._wandb = wandb.init(
                    project=cfg["logging"].get("project", "quadruped-rl-thesis"),
                    name=run_name,
                    config=cfg,
                    dir=str(self.run_dir),
                )
            except Exception as e:  # offline machines: degrade gracefully
                print(f"[logging] W&B unavailable ({e}); using local JSONL only")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {"step": step, "time": time.time(), **metrics}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()
