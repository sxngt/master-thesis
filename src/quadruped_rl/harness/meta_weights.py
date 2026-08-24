"""Meta-optimization of hybrid reward weights (alpha, beta, gamma).

Outer loop: sample weight vectors, run short inner training, score by the
target KPI, refine with Optuna TPE. Keeps weight search decoupled from the
main training loop (configs/reward/hybrid_llm.yaml holds the init values).
"""

from __future__ import annotations

import optuna

from quadruped_rl.harness.config import deep_merge
from quadruped_rl.harness.trainer import Trainer


def optimize_weights(
    base_cfg: dict,
    n_trials: int = 20,
    inner_timesteps: int = 5_000_000,
    objective_metric: str = "success_rate",
) -> dict[str, float]:
    def objective(trial: optuna.Trial) -> float:
        weights = {
            "alpha": trial.suggest_float("alpha", 0.5, 1.5),
            "beta": trial.suggest_float("beta", 0.0, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }
        cfg = deep_merge(
            base_cfg,
            {
                "reward": weights,
                "run": {"total_timesteps": inner_timesteps},
            },
        )
        result = Trainer(cfg).train()
        return float(result["final"].get(objective_metric, 0.0))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
