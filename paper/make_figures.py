#!/usr/bin/env python
"""학위논문 첨부용 그림·표·실험 이력 생성기.

출력:
  paper/figures/fig{N}_*.png|.pdf  — 그림 속 글 영문, 캡션은 본문에서 부여
  paper/tables/table{N}_*.csv      — 표 영문(제목 상단은 본문에서), 논문 규정 준수
  paper/tables/run_history.csv     — 전체 학습 이력(부록용)

실행: uv run python paper/make_figures.py
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

ALGOS = ["ppo", "trpo", "sac", "td3"]
TERRAINS = ["flat", "stairs", "rough"]
TLABEL = {"flat": "Flat", "stairs": "Stairs", "rough": "Rough"}
COLOR = {"ppo": "#2166ac", "trpo": "#67a9cf", "sac": "#b2182b", "td3": "#ef8a62"}

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
W = 5.3  # inches ~ 13.5 cm text width of 19 cm page


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig  {name}")


# ------------------------------------------------------------------ load runs
def load_runs():
    rows, curves = [], {}
    for run in sorted((ROOT / "data/results").glob("*_a1_*_s*_2026*")):
        m = re.match(r".+_(\d{8}-\d{6})_", run.name)
        if not m or not ("20260901-183400" <= m.group(1) < "20260903-014000"):
            continue  # core A1 baseline batches only
        cfg = yaml.safe_load((run / "config.yaml").read_text())
        algo, terrain = cfg["algorithm"]["name"], cfg["terrain"]["name"]
        seed = cfg["run"]["seed"]
        noise = cfg["algorithm"].get("noise_type", "")
        steps, vels, times, final, t0, t_last = [], [], [], {}, None, None
        for line in (run / "metrics.jsonl").read_text().splitlines():
            r = json.loads(line)
            t0 = t0 or r["time"]
            t_last = r["time"]
            if "eval/mean_forward_velocity_ms" in r:
                steps.append(r["step"])
                vels.append(r["eval/mean_forward_velocity_ms"])
                times.append(r["time"] - t0)
            f = {k[6:]: v for k, v in r.items() if k.startswith("final/")}
            if f:
                final = f
        if not final:
            continue
        rows.append({"algorithm": algo, "terrain": terrain, "seed": seed,
                     "noise": noise, "wall_min": (t_last - t0) / 60,
                     "budget": steps[-1] if steps else 0, **final})
        if algo in ALGOS:
            curves.setdefault((terrain, algo), []).append(
                (np.array(steps, float), np.array(vels)))
    return pd.DataFrame(rows), curves


df, curves = load_runs()
core = df[df.algorithm.isin(ALGOS)]
print(f"loaded {len(df)} runs ({len(core)} core)")

# =============================================================== Figure 1: curves
with plt.rc_context(STYLE):
    fig, axes = plt.subplots(1, 3, figsize=(W * 1.35, 2.6), sharey=True)
    for ax, t in zip(axes, TERRAINS):
        for a in ALGOS:
            seeds = curves.get((t, a), [])
            if not seeds:
                continue
            grid = seeds[0][0]
            stack = np.stack([np.interp(grid, s, v) for s, v in seeds])
            ax.plot(grid, stack.mean(0), color=COLOR[a], lw=1.4, label=a.upper())
            ax.fill_between(grid, stack.min(0), stack.max(0), color=COLOR[a], alpha=0.15)
        ax.axhline(0.8, color="gray", ls="--", lw=0.7)
        ax.set_xscale("log")
        ax.set_title(TLABEL[t])
        ax.set_xlabel("Environment steps")
    axes[0].set_ylabel("Forward velocity (m/s)")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "fig1_learning_curves")

# ============================================== Figure 2: velocity & success bars
agg = core.groupby(["terrain", "algorithm"]).agg(
    v=("mean_forward_velocity_ms", "mean"), v_sd=("mean_forward_velocity_ms", "std"),
    succ=("success_rate", "mean")).reset_index()
with plt.rc_context(STYLE):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W * 1.25, 2.5))
    x = np.arange(len(TERRAINS))
    bw = 0.19
    for i, a in enumerate(ALGOS):
        sub = agg[agg.algorithm == a].set_index("terrain").reindex(TERRAINS)
        ax1.bar(x + (i - 1.5) * bw, sub.v, bw, yerr=sub.v_sd, capsize=2,
                color=COLOR[a], label=a.upper(), error_kw={"lw": 0.8})
        ax2.bar(x + (i - 1.5) * bw, sub.succ * 100, bw, color=COLOR[a])
    ax1.axhline(1.0, color="gray", ls="--", lw=0.7)
    ax1.set_xticks(x, [TLABEL[t] for t in TERRAINS])
    ax2.set_xticks(x, [TLABEL[t] for t in TERRAINS])
    ax1.set_ylabel("Forward velocity (m/s)")
    ax2.set_ylabel("Success rate (%)")
    ax1.legend(frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(1.1, 1.02))
    save(fig, "fig2_velocity_success")

# ==================================================== Figure 3: reliability (seed SD)
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(W * 0.75, 2.4))
    sd = core.pivot_table(index="algorithm", columns="terrain",
                          values="mean_forward_velocity_ms", aggfunc="std")
    x = np.arange(len(TERRAINS))
    for i, a in enumerate(ALGOS):
        ax.bar(x + (i - 1.5) * 0.19, sd.loc[a, TERRAINS], 0.19,
               color=COLOR[a], label=a.upper())
    ax.set_xticks(x, [TLABEL[t] for t in TERRAINS])
    ax.set_ylabel("Seed SD of velocity (m/s)")
    ax.legend(frameon=False, ncol=2)
    save(fig, "fig3_seed_reliability")

# ================================= Figure 4: sample vs wall-clock efficiency (flat)
eff = pd.read_csv(ROOT / "data/results/learning_efficiency.csv")
eff_flat = eff[eff.terrain == "flat"].groupby("algorithm").agg(
    s2v=("s2v", "mean"), t2v=("t2v_min", "mean")).reindex(ALGOS)
with plt.rc_context(STYLE):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W * 1.1, 2.4))
    ax1.bar([a.upper() for a in ALGOS], eff_flat.s2v / 1e6,
            color=[COLOR[a] for a in ALGOS])
    ax1.set_yscale("log")
    ax1.set_ylabel("Steps to 0.8 m/s (millions)")
    ax2.bar([a.upper() for a in ALGOS], eff_flat.t2v,
            color=[COLOR[a] for a in ALGOS])
    ax2.set_ylabel("Wall-clock to 0.8 m/s (min)")
    save(fig, "fig4_learning_efficiency")

# ==================================================== Figure 5: DDPG decomposition
dd = df[(df.terrain == "flat")]
lab = [("ddpg", "ou", "DDPG\n(OU)"), ("ddpg", "parameter_space", "DDPG\n(param)"),
       ("td3", "", "TD3"), ("sac", "", "SAC")]
vals = []
for a, n, _ in lab:
    sub = dd[(dd.algorithm == a) & (dd.noise == n)] if a == "ddpg" else dd[dd.algorithm == a]
    vals.append(sub.mean_forward_velocity_ms.mean())
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.4))
    ax.bar([x[2] for x in lab], vals,
           color=["#999999", "#666666", COLOR["td3"], COLOR["sac"]])
    ax.axhline(0.8, color="gray", ls="--", lw=0.7)
    ax.set_ylabel("Forward velocity (m/s)")
    save(fig, "fig5_ddpg_failure_decomposition")

# ========================================================================= Tables
def sd0(x):
    return x.std(ddof=1) if len(x) > 1 else 0.0

t1 = core.groupby(["terrain", "algorithm"]).agg(
    Velocity=("mean_forward_velocity_ms", "mean"), Velocity_SD=("mean_forward_velocity_ms", sd0),
    Success=("success_rate", "mean"), CoT=("cost_of_transport", "mean"),
    Attitude=("attitude_stability", "mean"), Falls=("fall_frequency_per_min", "mean"),
    Path_Eff=("path_efficiency", "mean")).round(3).reset_index()
t1.columns = ["Terrain", "Algorithm", "Velocity (m/s)", "SD", "Success Rate",
              "Cost of Transport", "Attitude RMS (rad)", "Falls (/min)", "Path Efficiency"]
t1.to_csv(TAB / "table1_final_performance.csv", index=False)
print("  tab  table1_final_performance")

from quadruped_rl.analysis.statistics import compare_algorithms  # noqa: E402
rows = []
for t in TERRAINS:
    for metric in ["mean_forward_velocity_ms", "success_rate"]:
        sub = core[core.terrain == t]
        if sub.groupby("algorithm")[metric].std().max() < 1e-12:
            continue
        rep = compare_algorithms(sub, metric)
        a = rep["anova"]
        sig = rep["tukey"][rep["tukey"].reject.astype(str) == "True"]
        pairs = "; ".join(f"{r.group1.upper()}-{r.group2.upper()} (p={float(r['p-adj']):.3f}, d={abs(r.cohens_d):.1f})"
                          for _, r in sig.iterrows()) or "none"
        rows.append({"Terrain": TLABEL[t], "Metric": metric.replace("mean_forward_velocity_ms", "Velocity").replace("success_rate", "Success"),
                     "F": round(a["f_stat"], 2), "p": round(a["p_value"], 4),
                     "Eta Squared": round(a["eta_squared"], 2), "Significant Pairs (Tukey HSD)": pairs})
pd.DataFrame(rows).to_csv(TAB / "table2_statistics.csv", index=False)
print("  tab  table2_statistics")

t3 = eff[eff.algorithm.isin(ALGOS)].groupby(["terrain", "algorithm"]).agg(
    Steps_M=("s2v", lambda x: round(x.mean() / 1e6, 2)),
    Censored=("s2v", lambda x: f"{x.isna().sum()}/3"),
    Wall_Min=("t2v_min", lambda x: round(x.mean(), 1)),
    Throughput=("sps", lambda x: round(x.mean())),
    AUC=("auc", lambda x: round(x.mean(), 2))).reset_index()
t3.columns = ["Terrain", "Algorithm", "Steps to 0.8 m/s (M)", "Censored",
              "Wall-Clock (min)", "Throughput (steps/s)", "Normalized AUC"]
t3.to_csv(TAB / "table3_learning_efficiency.csv", index=False)
print("  tab  table3_learning_efficiency")

# ------------------------------------------------------------ run history (부록)
hist = df.sort_values(["terrain", "algorithm", "noise", "seed"])[
    ["algorithm", "noise", "terrain", "seed", "budget", "wall_min",
     "mean_forward_velocity_ms", "success_rate", "cost_of_transport",
     "attitude_stability", "fall_frequency_per_min"]].round(3)
hist.columns = ["Algorithm", "Noise", "Terrain", "Seed", "Budget (steps)",
                "Wall Time (min)", "Velocity (m/s)", "Success", "CoT",
                "Attitude RMS", "Falls (/min)"]
hist.to_csv(TAB / "run_history.csv", index=False)
print(f"  tab  run_history ({len(hist)} runs)")
print("done.")
