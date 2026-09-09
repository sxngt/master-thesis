#!/usr/bin/env python
"""학위논문 부록 표 생성기.

출력(paper/tables/):
  appA_coach_tunables.csv     코치 조정 파라미터·범위·척도 (configs/coach/llm_local.yaml)
  appB_hyperparameters.csv    알고리즘별 하이퍼파라미터 (configs/algorithm/*.yaml)
  appC_coach_runs.csv         제2부 실행별 최종 성적 (evolve4 control/v7p, ablation_v7p random/hillclimb)
  appD_coach_history.csv      코치 내부 개정 이력 (table9_coach_versions.csv의 축약)
실행: uv run python paper/make_appendix_tables.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "paper" / "tables"
sys.path.insert(0, str(ROOT / "paper"))
from make_coach_figures import load_runs  # noqa: E402

ARMS = [  # (label, run root)
    ("PPO", "data/results/evolve4/control-none/runs"),
    ("LLM-PPO (v3)", "data/results/evolve4/v7p/runs"),
    ("Random", "data/results/ablation_v7p/random/runs"),
    ("Hill-climbing", "data/results/ablation_v7p/hillclimb/runs"),
]
SETTING_LABEL = {
    "rough-hard-traditional": "Rough (hard)",
    "stairs-medium-traditional": "Stairs (medium)",
    "rough-medium-naive": "Rough (medium), naive",
}
# (yaml key path, table label); values are rendered as in the YAML
HP_KEYS = [
    ("learning_rate", "Learning rate"),
    ("gamma", "Discount γ"),
    ("gae_lambda", "GAE λ"),
    ("clip_range", "Clip range ε"),
    ("adaptive_kl.target_kl", "KL target"),
    ("adaptive_kl.penalty_min", "KL β min"),
    ("adaptive_kl.penalty_max", "KL β max"),
    ("adaptive_kl.stop_kl", "KL early stop"),
    ("entropy_coef", "Entropy coef"),
    ("value_coef", "Value coef"),
    ("max_grad_norm", "Grad-norm clip"),
    ("rollout_steps", "Rollout steps / env"),
    ("num_epochs", "Epochs"),
    ("num_minibatches", "Minibatches"),
    ("max_kl", "Max KL δ"),
    ("cg_iters", "CG iterations"),
    ("cg_damping", "CG damping"),
    ("line_search_steps", "Line-search steps"),
    ("line_search_backtrack", "Line-search backtrack"),
    ("value_lr", "Value LR"),
    ("num_workers", "Workers"),
    ("gradient_accumulation", "Grad accumulation"),
    ("tau", "Target τ"),
    ("init_temperature", "Init temperature α"),
    ("auto_temperature", "Auto temperature"),
    ("double_q", "Double Q"),
    ("clipped_double_q", "Clipped double Q"),
    ("policy_delay", "Policy delay"),
    ("target_noise", "Target noise"),
    ("noise_clip", "Target noise clip"),
    ("exploration_noise", "Exploration noise"),
    ("ou_theta", "OU θ"),
    ("ou_sigma", "OU σ"),
    ("param_noise_stddev", "Param-noise σ"),
    ("buffer_size", "Replay buffer"),
    ("batch_size", "Batch size"),
    ("warmup_steps", "Warm-up steps"),
    ("steps_per_iteration", "Steps / iteration"),
    ("updates_per_step", "Updates / step"),
    ("network.actor.hidden", "Actor hidden"),
    ("network.critic.hidden", "Critic hidden"),
    ("network.actor.activation", "Activation"),
]
# batch overrides applied on top of the YAML defaults in every off-policy run
HP_OVERRIDES = {"updates_per_step": {"sac": 16, "td3": 16, "ddpg": 16}}


def _get(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, list):
        return "-".join(str(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def tunables() -> None:
    cfg = yaml.safe_load((ROOT / "configs/coach/llm_local.yaml").read_text())["coach"]
    rows = []
    for name, spec in cfg["params"].items():
        rows.append(
            {
                "Parameter": name,
                "Lower": _fmt(spec["low"]),
                "Upper": _fmt(spec["high"]),
                "Scale": spec.get("scale", "linear"),
                "Per-Intervention Limit": "×/÷3" if spec.get("scale") == "log" else "±30%",
                "Curriculum Lever": "yes" if name in cfg["curriculum_params"] else "no",
                "Description": spec.get("description", ""),
            }
        )
    _write("appA_coach_tunables.csv", rows)


def hyperparameters() -> None:
    algos = ["ppo", "trpo", "a3c", "sac", "td3", "ddpg"]
    cfgs = {
        a: yaml.safe_load((ROOT / f"configs/algorithm/{a}.yaml").read_text())["algorithm"]
        for a in algos
    }
    rows = []
    for path, label in HP_KEYS:
        vals = {}
        for a in algos:
            v = HP_OVERRIDES.get(path, {}).get(a, _get(cfgs[a], path))
            if path == "network.actor.activation" and v is not None:
                v = f"{v}/{_get(cfgs[a], 'network.critic.activation')}"
            vals[a] = v
        if all(v is None for v in vals.values()):
            continue
        rows.append({"Hyperparameter": label, **{a.upper(): _fmt(vals[a]) for a in algos}})
    _write("appB_hyperparameters.csv", rows)


def coach_runs() -> None:
    rows = []
    for label, root in ARMS:
        p = ROOT / root
        if not p.exists():
            print(f"skip {root} (missing)")
            continue
        for r in load_runs(p):
            first = r["first_success50_Msteps"]
            rows.append(
                {
                    "Condition": label,
                    "Setting": SETTING_LABEL.get(r["setting"], r["setting"]),
                    "Seed": r["seed"],
                    "Objective J": f"{r['objective']:.3f}",
                    "Success": f"{r['success_rate']:.3f}",
                    "Velocity (m/s)": f"{r['mean_forward_velocity_ms']:.3f}",
                    "Falls (/min)": f"{r['fall_frequency_per_min']:.1f}",
                    "CoT": f"{r['cost_of_transport']:.3f}",
                    "Steps to Success 0.5 (M)": "—" if math.isnan(first) else f"{first:.0f}",
                    "Interventions": len(r["_interventions"]),
                    "Escaped": "yes" if r["escaped"] else "no",
                }
            )
    order = {SETTING_LABEL[k]: i for i, k in enumerate(SETTING_LABEL)}
    cond = {label: i for i, (label, _) in enumerate(ARMS)}
    rows.sort(key=lambda x: (order.get(x["Setting"], 9), cond[x["Condition"]], x["Seed"]))
    _write("appC_coach_runs.csv", rows)


def coach_history() -> None:
    """Nine columns: parent and comparison target are folded into Change / Note."""
    rows = []
    with (TAB / "table9_coach_versions.csv").open() as f:
        for r in csv.DictReader(f):
            dj = r["dJ_mean"] if r["dJ_sd"] in ("-", "") else f"{r['dJ_mean']}±{r['dJ_sd']}"
            change, note = r["change"], r["note"]
            if r["parent"] not in ("-", ""):
                change = f"from {r['parent']}: {change}"
            if r["compared_with"] != r["parent"]:
                note = f"vs {r['compared_with']}; {note}"
            rows.append(
                {
                    "Version": r["version"],
                    "Gen.": r["generation"].replace("meta-LLM", "LLM"),
                    "Change": change,
                    "n": r["n_pairs"],
                    "ΔJ (mean±SD)": dj,
                    "p": r["p_one_sided"].replace(" (Wilcoxon ", "; W ").replace(")", ""),
                    "d": r["cohens_d"],
                    "Verdict": r["verdict"],
                    "Note": note,
                }
            )
    _write("appD_coach_history.csv", rows)


def _write(name: str, rows: list[dict]) -> None:
    out = TAB / name
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.relative_to(ROOT)} ({len(rows)} rows)")


if __name__ == "__main__":
    tunables()
    hyperparameters()
    coach_runs()
    coach_history()
