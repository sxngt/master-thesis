#!/usr/bin/env python
"""학위논문 제2부(LLM 보상 코치) 첨부용 표·그림 생성기.

입력: 진화 루프 결과 디렉터리(data/results/evolve*/<arm>/runs, 각 런의 config.yaml·
metrics.jsonl·coach_log.jsonl). 코치 팔은 (설정, 시드)로 대조군과 짝짓는다.
출력:
  tables/table6_coach_per_setting.csv  설정별 PPO vs 코치, 항목별 짝 Δ·p·d (n=3)
  tables/table7_coach_pooled.csv       3 설정 통합(n=9): Δ, 95% CI, 짝 t·Wilcoxon p, d
  tables/table10_naive_recipe.csv      최초 보고 조리법 × naive 바닥 탈출 (Fisher)
  figures/fig7_coach_objective.*       설정별 J 학습곡선, PPO vs 코치, 시드별
  figures/fig8_naive_recovery.*        naive 성공률 곡선 + 코치 개입 표시
실행: uv run python paper/make_coach_figures.py \
        --control data/results/evolve3/control-none/runs \
        --coach data/results/evolve3/v7p/runs --coach-label LLM-PPO \
        --recipe-roots data/results/evolve3/v7/runs data/results/evolve3/v7p/runs ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
W = 5.3
STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
}
SETTINGS = ["rough-hard-traditional", "stairs-medium-traditional", "rough-medium-naive"]
SLABEL = {
    "rough-hard-traditional": "Rough (hard)",
    "stairs-medium-traditional": "Stairs (medium)",
    "rough-medium-naive": "Rough (medium), naive reward",
}
METRICS = [  # (final/ key, table label, better)
    ("objective", "Objective J", "+"),
    ("success_rate", "Success Rate", "+"),
    ("mean_forward_velocity_ms", "Forward Velocity (m/s)", "+"),
    ("cost_of_transport", "Cost of Transport", "-"),
    ("fall_frequency_per_min", "Falls (/min)", "-"),
    ("path_efficiency", "Path Efficiency", "+"),
    ("attitude_stability", "Attitude RMS (rad)", "-"),
    ("first_success50_Msteps", "Steps to Success 0.5 (M)", "-"),
]
WEIGHTS = {"success_rate": 1.0, "mean_forward_velocity_ms": 0.5}
FLOOR = 0.1  # playbook.FLOOR_SUCCESS: a policy has "left the floor" above this


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_run(d: Path) -> dict | None:
    cfg_p, met_p = d / "config.yaml", d / "metrics.jsonl"
    if not (cfg_p.exists() and met_p.exists()):
        return None
    cfg = yaml.safe_load(cfg_p.read_text())
    recs = _jsonl(met_p)
    evals = [r for r in recs if "eval/success_rate" in r]
    finals = [r for r in recs if "final/success_rate" in r]
    if not finals:
        return None
    fin = finals[-1]
    steps = np.array([r["step"] for r in evals], float)
    succ = np.array([r["eval/success_rate"] for r in evals], float)
    vel = np.array([r["eval/mean_forward_velocity_ms"] for r in evals], float)
    hit = np.nonzero(succ >= 0.5)[0]
    row = {
        "run": d.name,
        "setting": f"{cfg['terrain']['name']}-{cfg['sim'].get('terrain_level', 'easy')}-"
        f"{cfg.get('reward', {}).get('name', 'traditional')}",
        "seed": int(cfg["run"]["seed"]),
        "condition": (cfg.get("coach") or {}).get("strategy", "none"),
        "objective": sum(w * fin[f"final/{k}"] for k, w in WEIGHTS.items()),
        "first_success50_Msteps": steps[hit[0]] / 1e6 if len(hit) else math.nan,
        "escaped": bool((succ >= FLOOR).any()),
        "_steps": steps,
        "_J": succ + 0.5 * vel,
        "_succ": succ,
    }
    for k, v in fin.items():
        if k.startswith("final/"):
            row[k[6:]] = v
    log = _jsonl(d / "coach_log.jsonl")
    settled: dict[int, dict] = {}
    for rec in log:
        if rec["status"] != "pending" or rec["k"] not in settled:
            settled[rec["k"]] = rec
    row["_interventions"] = [r for r in settled.values() if r["status"] != "noop"]
    return row


def load_runs(root: Path) -> list[dict]:
    return [r for r in (load_run(d) for d in sorted(root.iterdir()) if d.is_dir()) if r]


def paired(ctrl: list[dict], coach: list[dict], key: str) -> dict:
    ref = {(r["setting"], r["seed"]): r for r in ctrl}
    a, b = [], []
    for r in coach:
        c = ref.get((r["setting"], r["seed"]))
        if c is None:
            continue
        x, y = c.get(key, math.nan), r.get(key, math.nan)
        if math.isnan(x) or math.isnan(y):  # censored (never reached the threshold)
            continue
        a.append(x)
        b.append(y)
    a, b = np.array(a), np.array(b)
    d = b - a
    n = len(d)
    out = {
        "n": n,
        "ctrl_mean": a.mean() if n else math.nan,
        "ctrl_sd": a.std(ddof=1) if n > 1 else math.nan,
        "coach_mean": b.mean() if n else math.nan,
        "coach_sd": b.std(ddof=1) if n > 1 else math.nan,
        "delta": d.mean() if n else math.nan,
        "ci95": math.nan,
        "p_t": math.nan,
        "p_w": math.nan,
        "d": math.nan,
    }
    if n > 1 and d.std(ddof=1) > 0:
        sd = d.std(ddof=1)
        out["ci95"] = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
        out["p_t"] = stats.ttest_rel(b, a).pvalue
        out["d"] = d.mean() / sd
        if n >= 6:
            out["p_w"] = stats.wilcoxon(b, a).pvalue
    return out


def fmt(x: float, nd: int = 3) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def table6(ctrl, coach, label) -> pd.DataFrame:
    rows = []
    for s in SETTINGS:
        c = [r for r in ctrl if r["setting"] == s]
        k = [r for r in coach if r["setting"] == s]
        if not c or not k:
            continue
        for key, name, _ in METRICS:
            p = paired(c, k, key)
            rows.append(
                {
                    "Setting": SLABEL[s],
                    "Metric": name,
                    "n": p["n"],
                    "PPO": f"{fmt(p['ctrl_mean'])} ± {fmt(p['ctrl_sd'])}",
                    label: f"{fmt(p['coach_mean'])} ± {fmt(p['coach_sd'])}",
                    "Paired Δ": fmt(p["delta"]),
                    "p (paired t)": fmt(p["p_t"]),
                    "Cohen's d": fmt(p["d"], 2),
                }
            )
            if key == "first_success50_Msteps":
                rows[-1]["PPO"] = " / ".join(
                    fmt(r["first_success50_Msteps"], 0) or "—"
                    for r in sorted(c, key=lambda r: r["seed"])
                )
                rows[-1][label] = " / ".join(
                    fmt(r["first_success50_Msteps"], 0) or "—"
                    for r in sorted(k, key=lambda r: r["seed"])
                )
    return pd.DataFrame(rows)


def table7(ctrl, coach, label) -> pd.DataFrame:
    rows = []
    for key, name, _ in METRICS:
        p = paired(ctrl, coach, key)
        rows.append(
            {
                "Metric": name,
                "n pairs": p["n"],
                "PPO": f"{fmt(p['ctrl_mean'])} ± {fmt(p['ctrl_sd'])}",
                label: f"{fmt(p['coach_mean'])} ± {fmt(p['coach_sd'])}",
                "Paired Δ": fmt(p["delta"]),
                "95% CI": f"± {fmt(p['ci95'])}" if not math.isnan(p["ci95"]) else "",
                "p (paired t)": fmt(p["p_t"]),
                "p (Wilcoxon)": fmt(p["p_w"]),
                "Cohen's d": fmt(p["d"], 2),
            }
        )
    return pd.DataFrame(rows)


def recipe_of(r: dict) -> tuple[bool, bool, str]:
    """First-report intervention: (target_ms down, energy penalty weakened, third key)."""
    iv = r["_interventions"]
    if not iv:
        return False, False, ""
    first = iv[0]
    before, applied = first.get("params_before", {}), first.get("applied", {})
    down = applied.get("forward_velocity.target_ms", 9) < before.get(
        "forward_velocity.target_ms", 0
    )
    weaker = applied.get("energy.weight", -9) > before.get("energy.weight", 0)  # toward 0
    other = [k for k in applied if k not in ("forward_velocity.target_ms", "energy.weight")]
    return down, weaker, ",".join(other)


def table10(roots: list[Path]) -> pd.DataFrame:
    runs = [r for root in roots for r in load_runs(root) if r["setting"].endswith("naive")]
    a = [r for r in runs if all(recipe_of(r)[:2])]
    b = [r for r in runs if not all(recipe_of(r)[:2])]
    ea, eb = sum(r["escaped"] for r in a), sum(r["escaped"] for r in b)
    p = (
        stats.fisher_exact([[ea, len(a) - ea], [eb, len(b) - eb]], alternative="greater").pvalue
        if a and b
        else math.nan
    )
    df = pd.DataFrame(
        [
            {
                "First-Report Recipe": "target_ms ↓ and energy penalty ↓",
                "Runs": len(a),
                "Escaped Floor": ea,
                "Escape Rate": fmt(ea / len(a) if a else math.nan, 2),
                "Fisher p (one-sided)": fmt(p),
            },
            {
                "First-Report Recipe": "other / none",
                "Runs": len(b),
                "Escaped Floor": eb,
                "Escape Rate": fmt(eb / len(b) if b else math.nan, 2),
                "Fisher p (one-sided)": "",
            },
        ]
    )
    return df


def fig7(ctrl, coach, label, suffix=""):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(W * 1.35, 2.3), sharey=True)
    for ax, s in zip(axes, SETTINGS, strict=True):
        for r in sorted([r for r in ctrl if r["setting"] == s], key=lambda r: r["seed"]):
            ax.plot(
                r["_steps"] / 1e6,
                r["_J"],
                color="#888888",
                lw=0.9,
                ls="--",
                label="PPO" if r["seed"] == 0 else None,
            )
        for r in sorted([r for r in coach if r["setting"] == s], key=lambda r: r["seed"]):
            ax.plot(
                r["_steps"] / 1e6,
                r["_J"],
                color="#2166ac",
                lw=1.0,
                label=label if r["seed"] == 0 else None,
            )
        ax.set_title(SLABEL[s])
        ax.set_xlabel("Environment steps (M)")
    axes[0].set_ylabel("Objective J")
    axes[0].legend(loc="lower right", frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig7_coach_objective{suffix}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig8(ctrl, coach, label, suffix=""):
    plt.rcParams.update(STYLE)
    s = "rough-medium-naive"
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for r in sorted([r for r in ctrl if r["setting"] == s], key=lambda r: r["seed"]):
        ax.plot(
            r["_steps"] / 1e6,
            r["_succ"],
            color="#888888",
            lw=0.9,
            ls="--",
            label="PPO (all seeds)" if r["seed"] == 0 else None,
        )
    colors = ["#2166ac", "#4393c3", "#92c5de"]
    for r in sorted([r for r in coach if r["setting"] == s], key=lambda r: r["seed"]):
        c = colors[r["seed"] % 3]
        ax.plot(r["_steps"] / 1e6, r["_succ"], color=c, lw=1.0, label=f"{label} seed {r['seed']}")
        for iv in r["_interventions"]:
            if iv["status"] == "pending":  # applied at the last report, never judged
                continue
            m = "|" if iv["status"] == "kept" else "x"
            ax.plot(iv["step"] / 1e6, -0.05 - 0.04 * (r["seed"] % 3), marker=m, color=c, ms=4, lw=0)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Success rate")
    ax.legend(loc="center right", frameon=False)
    ax.text(
        0.01,
        0.02,
        "markers: coach interventions (| kept, × rolled back)",
        transform=ax.transAxes,
        fontsize=7,
    )
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig8_naive_recovery{suffix}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--control", required=True, help="runs dir of the no-coach arm")
    p.add_argument("--coach", required=True, help="runs dir of the coached arm")
    p.add_argument("--coach-label", default="LLM-PPO")
    p.add_argument("--recipe-roots", nargs="*", default=[], help="coached runs dirs for table 10")
    p.add_argument("--suffix", default="", help="output-name suffix, e.g. _evolve4 (default: none)")
    a = p.parse_args()
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    ctrl, coach = load_runs(ROOT / a.control), load_runs(ROOT / a.coach)
    print(f"control {len(ctrl)} runs, {a.coach_label} {len(coach)} runs")
    t6, t7 = table6(ctrl, coach, a.coach_label), table7(ctrl, coach, a.coach_label)
    t6.to_csv(TAB / f"table6_coach_per_setting{a.suffix}.csv", index=False)
    t7.to_csv(TAB / f"table7_coach_pooled{a.suffix}.csv", index=False)
    print(t7.to_string(index=False))
    if a.recipe_roots:
        t10 = table10([ROOT / r for r in a.recipe_roots])
        t10.to_csv(TAB / f"table10_naive_recipe{a.suffix}.csv", index=False)
        print(t10.to_string(index=False))
    fig7(ctrl, coach, a.coach_label, a.suffix)
    fig8(ctrl, coach, a.coach_label, a.suffix)
    print("saved tables 6, 7, 10 and figures 7, 8")


if __name__ == "__main__":
    main()
