"""Hyperparameter optimization via Optuna Bayesian search (TPE).

Search spaces live in configs/algorithm/<name>.yaml (search_space section);
this module only interprets them, keeping tuning declarative.
"""

from __future__ import annotations

from typing import Any

import optuna

from quadruped_rl.harness.config import compose_config, deep_merge
from quadruped_rl.harness.trainer import Trainer


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    if "choices" in spec:
        return trial.suggest_categorical(name, spec["choices"])
    if (
        isinstance(spec.get("low"), int)
        and isinstance(spec.get("high"), int)
        and not spec.get("log")
    ):
        return trial.suggest_int(name, spec["low"], spec["high"])
    return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))


def run_sweep(
    algorithm: str,
    robot: str,
    terrain: str,
    n_trials: int = 50,
    objective_metric: str = "success_rate",
    timesteps: int | None = None,
    storage: str | None = None,
) -> optuna.Study:
    base = compose_config(algorithm=algorithm, robot=robot, terrain=terrain)
    space = base.get("search_space", {})

    def objective(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, spec) for name, spec in space.items()}
        overrides: dict[str, Any] = {"algorithm": params}
        if timesteps:
            overrides["run"] = {"total_timesteps": timesteps}
        cfg = deep_merge(base, overrides)
        result = Trainer(cfg).train()
        return float(result["final"].get(objective_metric, 0.0))

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=f"{algorithm}_{robot}_{terrain}",
        sampler=optuna.samplers.TPESampler(seed=0),
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)
    return study
