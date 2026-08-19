"""Experiment matrix runner: expands an experiment config into the full
(robot x algorithm x terrain x seed) grid and executes runs sequentially
or via generated job scripts for a cluster.

Phase 2: 3 robots x 6 algorithms x 12 terrains x 10 seeds = 2,160 runs.
"""

from __future__ import annotations

import itertools
import json
import traceback
from pathlib import Path
from typing import Any

from quadruped_rl.harness.config import compose_config
from quadruped_rl.harness.trainer import DATA_ROOT, Trainer


def expand_matrix(exp_cfg: dict[str, Any]) -> list[dict[str, str | int]]:
    e = exp_cfg["experiment"]
    return [
        {"robot": r, "algorithm": a, "terrain": t, "seed": s}
        for r, a, t, s in itertools.product(e["robots"], e["algorithms"], e["terrains"], e["seeds"])
    ]


class MatrixRunner:
    def __init__(self, experiment: str, overrides: dict | None = None):
        self.experiment = experiment
        self.overrides = overrides or {}
        self.exp_cfg = compose_config(experiment=experiment)
        self.status_path = DATA_ROOT / "results" / f"{experiment}_status.json"
        self.status: dict[str, Any] = self._load_status()

    def _load_status(self) -> dict[str, Any]:
        if self.status_path.exists():
            return json.loads(self.status_path.read_text())
        return {}

    def _save_status(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(self.status, indent=2))

    @staticmethod
    def cell_key(cell: dict[str, Any]) -> str:
        return f"{cell['algorithm']}|{cell['robot']}|{cell['terrain']}|s{cell['seed']}"

    def run(self, resume: bool = True) -> None:
        cells = expand_matrix(self.exp_cfg)
        reward = self.exp_cfg["experiment"].get("reward", "traditional")
        print(f"[matrix] {self.experiment}: {len(cells)} cells")
        for i, cell in enumerate(cells):
            key = self.cell_key(cell)
            if resume and self.status.get(key, {}).get("state") == "done":
                continue
            print(f"[matrix] ({i + 1}/{len(cells)}) {key}")
            cfg = compose_config(
                algorithm=str(cell["algorithm"]),
                robot=str(cell["robot"]),
                terrain=str(cell["terrain"]),
                reward=reward,
                experiment=self.experiment,
                overrides={**self.overrides, "run": {"seed": cell["seed"]}},
            )
            try:
                result = Trainer(cfg).train()
                self.status[key] = {"state": "done", **result}
            except Exception as e:
                self.status[key] = {
                    "state": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            self._save_status()

    def export_jobs(self, out_path: str | Path) -> Path:
        """Write one CLI command per cell (for SLURM/GNU-parallel dispatch)."""
        cells = expand_matrix(self.exp_cfg)
        reward = self.exp_cfg["experiment"].get("reward", "traditional")
        lines = [
            "python scripts/train.py"
            f" --algorithm {c['algorithm']} --robot {c['robot']}"
            f" --terrain {c['terrain']} --reward {reward}"
            f" --experiment {self.experiment} --seed {c['seed']}"
            for c in cells
        ]
        out = Path(out_path)
        out.write_text("\n".join(lines) + "\n")
        return out
