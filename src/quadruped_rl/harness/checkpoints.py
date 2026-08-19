"""Checkpoint save/load with metadata for resumable, reproducible runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class CheckpointManager:
    def __init__(self, run_dir: str | Path, keep_last: int = 3):
        self.dir = Path(run_dir) / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._best_metric: float | None = None

    def save(
        self, algorithm, step: int, metrics: dict[str, Any], is_best_metric: str = "success_rate"
    ) -> Path:
        path = self.dir / f"step_{step:012d}.pt"
        algorithm.save(path)
        meta = {"step": step, "time": time.time(), "metrics": metrics}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        value = metrics.get(is_best_metric)
        if value is not None and (self._best_metric is None or value > self._best_metric):
            self._best_metric = value
            algorithm.save(self.dir / "best.pt")
            (self.dir / "best.json").write_text(json.dumps(meta, indent=2))

        self._prune()
        return path

    def _prune(self) -> None:
        ckpts = sorted(self.dir.glob("step_*.pt"))
        for old in ckpts[: -self.keep_last]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)
