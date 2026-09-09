"""Part 2 figures and tables around the three thesis versions of the reward coach.

The thesis collapses the internal iterations (v1 ... v13 in docs/coach_versions.md) into
three versions that changed the architecture or the outcome the most:

    thesis v1  = internal v3   fixed-threshold guard, unlimited best restore   (coach_v3, GPT API)
    thesis v2  = internal v5   statistical decision layer + release invariant  (evolve, local LLM)
    thesis v3  = internal v7p  ledger veto, curriculum lock, cross-run playbook (evolve4, local LLM)

Every version is compared with the coach-free PPO runs of its own batch (same seeds), so the
paired numbers are within-batch even though the stack moved between batches.

Outputs (paper/figures, paper/tables; thesis figure/table numbers are assigned by
paper/build_thesis.py in order of appearance):
    coach_architecture            block diagram, components coloured by the version that added them
    coach_version_lineage         internal lineage -> thesis versions, paired dJ forest plot
    coach_objective_by_version    J during training, PPO vs LLM-PPO, 3 versions x 3 settings
    coach_curriculum_lever        commanded speed on stairs per version, restores/rollbacks marked
    coach_naive_recovery          success rate under the naive reward per version
    coach_decision_activity       intervention outcomes per run, per version and setting
    coach_versions_thesis.csv, coach_intervention_stats.csv, coach_naive_recipe.csv

    uv run python paper/make_version_figures.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["font.serif"] = ["Liberation Serif", "Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"

sys.path.insert(0, str(Path(__file__).parent))
from make_coach_figures import (  # noqa: E402
    FIG,
    FLOOR,
    ROOT,
    SETTINGS,
    SLABEL,
    STYLE,
    TAB,
    W,
    _jsonl,
    fmt,
    load_runs,
    paired,
    recipe_of,
)
from scipy import stats  # noqa: E402

# every LLM-coached naive-reward run, grouped by LLM (v1 = GPT API; v2/v3 lineage = local LLM,
# including the rejected automated candidates that share the same decision layer)
RECIPE_GROUPS = [
    ("v1 (GPT-5.4 API)", [("data/results/coach_v3/runs", "llm")]),
    (
        "v2–v3 lineage (local Qwen3.8-27B)",
        [
            ("data/results/evolve_remote/v5/runs", None),
            ("data/results/evolve3/v7/runs", None),
            ("data/results/evolve3/v7p/runs", None),
            ("data/results/evolve3/v8/runs", None),
            ("data/results/evolve3/v9/runs", None),
            ("data/results/evolve4/v7p/runs", None),
            ("data/results/evolve4/v10/runs", None),
            ("data/results/evolve4/v11/runs", None),
            ("data/results/evolve4/v12/runs", None),
        ],
    ),
]

VERSIONS = [
    {
        "key": "v1",
        "label": "v1",
        "title": "v1: fixed-threshold guard",
        "internal": "v1–v3",
        "dates": "Sep 4–5",
        "llm": "GPT-5.4 (API)",
        "coach": ("data/results/coach_v3/runs", "llm"),
        "ctrl": ("data/results/coach_v3/runs", "none"),
        "added": (
            "range/sign/step guardrails; rollback at ΔJ < −0.05; best-snapshot restore; "
            "curriculum lever"
        ),
        "color": "#74a9cf",
    },
    {
        "key": "v2",
        "label": "v2",
        "title": "v2: statistical decision layer",
        "internal": "v4–v5",
        "dates": "Sep 5–8",
        "llm": "Qwen3.8-27B (local, 8-bit)",
        "coach": ("data/results/evolve_remote/v5/runs", "llm"),
        "ctrl": ("data/results/evolve_remote/control-none/runs", "none"),
        "added": "noise-aware τ from SE and process noise; confirm re-evaluation; restore ≤ 1; "
        "detrended effect ledger; phase schedule; curriculum release invariant; headroom",
        "color": "#2b8cbe",
    },
    {
        "key": "v3",
        "label": "v3",
        "title": "v3: ledger veto, lock, playbook",
        "internal": "v5p–v7p",
        "dates": "Sep 8–9",
        "llm": "Qwen3.8-27B (local, 4-bit)",
        "coach": ("data/results/evolve4/v7p/runs", "llm"),
        "ctrl": ("data/results/evolve4/control-none/runs", "none"),
        "added": "ledger veto; curriculum lock in the release phase; settled effect; "
        "cross-run playbook; synchronous coach; bounded KL β",
        "color": "#045a8d",
    },
]
PPO_COLOR = "#8c8c8c"
SEED_ALPHA = 0.45
LEVER = "forward_velocity.target_ms"

# internal lineage for fig8: (name, day offset from Sep 4, thesis version or None, verdict)
LINEAGE = [
    ("v1", 0.0, "v1", "iter"),
    ("v2", 0.7, "v1", "iter"),
    ("v3", 1.0, "v1", "measured"),
    ("v4", 1.5, "v2", "iter"),
    ("v5", 3.6, "v2", "measured"),
    ("v5p", 4.2, "v3", "iter"),
    ("v6", 4.4, None, "rejected"),
    ("v7", 4.5, "v3", "iter"),
    ("v7p", 4.8, "v3", "measured"),
    ("v8", 4.9, None, "rejected"),
    ("v9", 5.0, None, "rejected"),
    ("v10", 5.25, None, "rejected"),
    ("v11", 5.35, None, "rejected"),
    ("v12", 5.45, None, "rejected"),
    ("v13", 5.5, None, "not run"),
]


# ------------------------------------------------------------------ data
def load_version(v: dict) -> tuple[list[dict], list[dict]]:
    coach = [r for r in load_runs(ROOT / v["coach"][0]) if r["condition"] == v["coach"][1]]
    ctrl = [r for r in load_runs(ROOT / v["ctrl"][0]) if r["condition"] == v["ctrl"][1]]
    for r in coach:
        r["_lever"] = lever_trace(ROOT / v["coach"][0] / r["run"])
        r["_events"] = events(ROOT / v["coach"][0] / r["run"])
    return ctrl, coach


def lever_trace(d: Path) -> tuple[np.ndarray, np.ndarray]:
    """Commanded forward speed after every coach report (from metrics.jsonl)."""
    key = f"coach/param/{LEVER}"
    recs = [r for r in _jsonl(d / "metrics.jsonl") if key in r]
    return np.array([r["step"] for r in recs], float), np.array([r[key] for r in recs], float)


def events(d: Path) -> dict[str, list[int]]:
    """Steps of restores, rollbacks and layer refusals from the coach log (notes only)."""
    out: dict[str, list[int]] = {k: [] for k in ("restore", "rollback", "veto", "lock", "freeze")}
    seen: dict[int, dict] = {}
    for rec in _jsonl(d / "coach_log.jsonl"):
        if rec["status"] != "pending" or rec["k"] not in seen:
            seen[rec["k"]] = rec
    for rec in seen.values():
        if rec.get("restored_from_step"):
            out["restore"].append(rec["step"])
        if rec["status"] == "rolled_back":
            out["rollback"].append(rec["step"])
        for n in rec.get("notes") or []:
            if "vetoed by the ledger" in n:
                out["veto"].append(rec["step"])
            elif "locked in the release" in n:
                out["lock"].append(rec["step"])
            elif "frozen in the consolidate" in n:
                out["freeze"].append(rec["step"])
    return out


def _clipped(rec: dict) -> bool:
    """True if the guardrails altered or dropped part of the proposal."""
    prop, app = rec.get("proposed") or {}, rec.get("applied") or {}
    if set(prop) != set(app):
        return True
    return any(abs(prop[k] - app[k]) > 1e-6 * max(1.0, abs(prop[k])) for k in prop)


def per_run_stats(r: dict) -> dict:
    iv = r["_interventions"]
    settled = [i for i in iv if i["status"] != "pending"]
    steps, lever = r["_lever"]
    down = int(np.sum(np.diff(lever) < -1e-9)) if len(lever) > 1 else 0
    up = int(np.sum(np.diff(lever) > 1e-9)) if len(lever) > 1 else 0
    return {
        "interventions": len(iv),
        "kept": sum(i["status"] == "kept" for i in settled),
        "rolled_back": sum(i["status"] == "rolled_back" for i in settled),
        "restores": len(r["_events"]["restore"]),
        "refused": sum(len(r["_events"][k]) for k in ("veto", "lock", "freeze")),
        "clipped": sum(_clipped(i) for i in iv),
        "lever_down": down,
        "lever_up": up,
        "lever_final": float(lever[-1]) if len(lever) else math.nan,
    }


# ------------------------------------------------------------------ tables
def table9(data: dict[str, tuple[list[dict], list[dict]]]) -> pd.DataFrame:
    rows = []
    for v in VERSIONS:
        ctrl, coach = data[v["key"]]
        p = paired(ctrl, coach, "objective")
        naive = [r for r in coach if r["setting"] == "rough-medium-naive"]
        rows.append(
            {
                "Version": v["label"],
                "Internal Iterations": f"internal {v['internal']} ({v['dates']})",
                "LLM": v["llm"],
                "Decision Layer Added": v["added"],
                "n Pairs": p["n"],
                "PPO J": f"{fmt(p['ctrl_mean'])} ± {fmt(p['ctrl_sd'])}",
                "LLM-PPO J": f"{fmt(p['coach_mean'])} ± {fmt(p['coach_sd'])}",
                "Paired ΔJ": fmt(p["delta"]),
                "95% CI": f"± {fmt(p['ci95'])}",
                "p (paired t)": fmt(p["p_t"]),
                "p (Wilcoxon)": fmt(p["p_w"]),
                "Cohen's d": fmt(p["d"], 2),
                "Naive Escape": f"{sum(r['escaped'] for r in naive)} / {len(naive)}",
            }
        )
    cand = _candidates()
    rows.append(
        {
            "Version": "automated candidates",
            "Internal Iterations": "internal " + ", ".join(cand["version"]) + " (Sep 8–9)",
            "LLM": "meta-LLM = coach LLM",
            "Decision Layer Added": "one prompt or threshold change each (max_params, "
            "intervention cap, release threshold, raise rule, noise_z)",
            "n Pairs": f"{len(cand)} × 9 vs incumbent",
            "PPO J": "",
            "LLM-PPO J": "",
            "Paired ΔJ": f"{cand['dJ_mean'].min():+.3f} … {cand['dJ_mean'].max():+.3f}",
            "95% CI": "",
            "p (paired t)": (
                f"{cand['p_one_sided'].min():.2f} … {cand['p_one_sided'].max():.2f} (one-sided)"
            ),
            "p (Wilcoxon)": "",
            "Cohen's d": f"{cand['cohens_d'].min():+.2f} … {cand['cohens_d'].max():+.2f}",
            "Naive Escape": "all rejected",
        }
    )
    return pd.DataFrame(rows)


def table_recipe() -> pd.DataFrame:
    """First-report recipe x escape from the naive-reward floor, per LLM group and pooled."""
    rows = []
    pooled: list[dict] = []
    for label, roots in RECIPE_GROUPS:
        runs = []
        for root, cond in roots:
            runs += [
                r
                for r in load_runs(ROOT / root)
                if r["setting"].endswith("naive") and (cond is None or r["condition"] == cond)
            ]
        pooled += runs
        rows += _recipe_rows(label, runs)
    rows += _recipe_rows("All LLM-coached runs", pooled)
    return pd.DataFrame(rows)


def _recipe_rows(label: str, runs: list[dict]) -> list[dict]:
    a = [r for r in runs if all(recipe_of(r)[:2])]
    b = [r for r in runs if not all(recipe_of(r)[:2])]
    ea, eb = sum(r["escaped"] for r in a), sum(r["escaped"] for r in b)
    p = (
        fmt(
            stats.fisher_exact([[ea, len(a) - ea], [eb, len(b) - eb]], alternative="greater").pvalue
        )
        if a and b
        else "—"
    )
    return [
        {
            "LLM": label,
            "First-Report Recipe": "target_ms ↓ and energy penalty ↓",
            "Runs": len(a),
            "Escaped Floor": ea,
            "Escape Rate": fmt(ea / len(a), 2) if a else "—",
            "Fisher p (one-sided)": p,
        },
        {
            "LLM": label,
            "First-Report Recipe": "other / none",
            "Runs": len(b),
            "Escaped Floor": eb,
            "Escape Rate": fmt(eb / len(b), 2) if b else "—",
            "Fisher p (one-sided)": "",
        },
    ]


def table11(data: dict[str, tuple[list[dict], list[dict]]]) -> pd.DataFrame:
    rows = []
    for v in VERSIONS:
        ctrl, coach = data[v["key"]]
        for s in SETTINGS:
            k = [r for r in coach if r["setting"] == s]
            c = [r for r in ctrl if r["setting"] == s]
            st = pd.DataFrame([per_run_stats(r) for r in k])
            first = [r["first_success50_Msteps"] for r in k]
            first_c = [r["first_success50_Msteps"] for r in c]
            n_settled = st["kept"].sum() + st["rolled_back"].sum()
            rows.append(
                {
                    "Version": v["label"],
                    "Setting": SLABEL[s],
                    "n": len(k),
                    "Int. / Run": fmt(st["interventions"].mean(), 1),
                    "Kept (%)": fmt(100 * st["kept"].sum() / max(1, n_settled), 0),
                    "RB / Run": fmt(st["rolled_back"].mean(), 1),
                    "RS / Run": fmt(st["restores"].mean(), 1),
                    "Clip / Run": fmt(st["clipped"].mean(), 1),
                    "Lever ↓ / ↑": (f"{st['lever_down'].mean():.1f} / {st['lever_up'].mean():.1f}"),
                    "Final Cmd (m/s)": (
                        f"{st['lever_final'].mean():.2f} ± {st['lever_final'].std(ddof=1):.2f}"
                    ),
                    "Steps to 0.5 (M), LLM-PPO / PPO": f"{_median(first)} / {_median(first_c)}",
                }
            )
    return pd.DataFrame(rows)


def _candidates() -> pd.DataFrame:
    """Rejected meta-LLM candidates from the hand-curated version table, numeric columns parsed."""
    t9 = pd.read_csv(TAB / "table9_coach_versions.csv")
    t9 = t9[t9["verdict"].str.startswith("rejected")].copy()
    for c in ("n_pairs", "dJ_mean", "dJ_sd", "ci95_halfwidth", "p_one_sided", "cohens_d"):
        t9[c] = pd.to_numeric(t9[c], errors="coerce")
    return t9


def _median(xs: list[float]) -> str:
    ok = [x for x in xs if not math.isnan(x)]
    if not ok:
        return f"— (0/{len(xs)})"
    return f"{np.median(ok):.0f} ({len(ok)}/{len(xs)})"


# ------------------------------------------------------------------ fig 7: architecture
def _box(ax, x, y, w, h, title, lines, edge, lw=1.1, fs=7.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc="white", ec=edge, lw=lw
        )
    )
    ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=fs + 1.3, weight="bold")
    yy = y + h - 6.2
    for text, ver in lines:
        col = {"v1": "#4d4d4d", "v2": VERSIONS[1]["color"], "v3": VERSIONS[2]["color"]}[ver]
        ax.plot([x + 1.7], [yy], marker="s", ms=3.0, color=col, lw=0, clip_on=False)
        ax.text(x + 3.2, yy, text, ha="left", va="center", fontsize=fs, color="#222222")
        yy -= 3.6


def _arrow(
    ax, p, q, text=None, color="#333333", lw=1.0, ls="-", tpos=0.5, toff=(0, 1.0), ha="center"
):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=1,
            shrinkB=1,
        )
    )
    if text:
        mx = p[0] + (q[0] - p[0]) * tpos + toff[0]
        my = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(
            mx,
            my,
            text,
            fontsize=6.4,
            ha=ha,
            va="bottom",
            color=color,
            bbox=dict(fc="white", ec="none", pad=0.6),
        )


def fig7() -> None:
    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(7.0, 5.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")
    c1, c2, c3 = "#4d4d4d", VERSIONS[1]["color"], VERSIONS[2]["color"]

    _box(
        ax,
        1,
        54,
        22,
        25,
        "PPO training loop",
        [
            ("Isaac Lab, 4,096 envs", "v1"),
            (r"reward $\Sigma\, w_i r_i(s,a)$", "v1"),
            ("lever: commanded speed", "v1"),
            (r"KL β bounded [$2^{-6}$, 8]", "v3"),
        ],
        edge=c1,
    )
    _box(
        ax,
        26,
        54,
        22,
        25,
        "Fixed evaluation",
        [
            (r"every $10^6$ steps", "v1"),
            ("256 envs, deterministic", "v1"),
            ("J = success + 0.5·speed", "v1"),
            (r"SE(J), process noise $\hat{\sigma}$", "v2"),
        ],
        edge=c1,
    )
    _box(
        ax,
        51,
        54,
        22,
        25,
        "Report",
        [
            ("KPIs, reward-term shares", "v1"),
            ("gait, PPO statistics", "v1"),
            (r"J trace, $\hat{\sigma}$, headroom", "v2"),
            ("effect ledger, phase", "v2"),
            ("cross-run playbook", "v3"),
        ],
        edge=c1,
    )
    _box(
        ax,
        76,
        54,
        22,
        25,
        "Proposer",
        [
            ("LLM → JSON: ≤3 params,", "v1"),
            ("diagnosis, predicted ΔJ", "v2"),
            ("random: 3 params", "v1"),
            ("hill-climb: (1+1)-ES", "v1"),
        ],
        edge=c1,
    )
    _box(
        ax,
        26,
        6,
        47,
        38,
        "Decision layer (shared by every proposer)",
        [
            ("guardrails: range, sign, per-step limit (±30 %, ×/÷3)", "v1"),
            ("rollback if ΔJ < −τ, τ = 0.05; best-snapshot restore", "v1"),
            ("noise-aware τ = min{max(0.05, 2√2σ), 0.3}; confirm re-eval", "v2"),
            ("restore ≤ 1 per snapshot; cool-down after rollback", "v2"),
            ("detrended effect ledger (normal–normal posterior)", "v2"),
            ("phases: explore → exploit → consolidate (reward frozen)", "v2"),
            ("curriculum release invariant (raise the lowered lever)", "v2"),
            ("ledger veto; curriculum lock in release; settled effect", "v3"),
        ],
        edge=c3,
        lw=1.5,
    )
    _box(
        ax,
        1,
        6,
        22,
        25,
        "Apply / rollback",
        [
            ("reward weights, lever,", "v1"),
            ("LR, entropy coefficient", "v1"),
            ("rollback = policy snapshot", "v1"),
            ("+ parameters restored", "v1"),
        ],
        edge=c1,
    )
    _box(
        ax,
        76,
        6,
        22,
        25,
        "Records",
        [
            ("coach_log.jsonl: report,", "v1"),
            ("raw response, verdict", "v1"),
            ("playbook.json indexed", "v3"),
            ("from finished runs", "v3"),
        ],
        edge=c1,
    )

    _arrow(ax, (23, 66.5), (26, 66.5), "policy", toff=(0, 1.0))
    _arrow(ax, (48, 66.5), (51, 66.5), "metrics", toff=(0, 1.0))
    _arrow(ax, (73, 66.5), (76, 66.5), "prompt", toff=(0, 1.0))
    _arrow(ax, (84, 54), (70, 44), "proposal", toff=(3.5, 0), ha="left")
    _arrow(ax, (37, 54), (37, 44), r"$J_k$, $\hat{\sigma}_k$", toff=(1.5, -1.5), ha="left")
    _arrow(ax, (26, 22), (23, 22), "verdict", toff=(0, 1.0))
    _arrow(ax, (12, 31), (12, 54), r"next $2{\times}10^6$ steps", toff=(-1.2, -1.5), ha="right")
    _arrow(ax, (73, 10), (76, 10), "log", color="#666666", toff=(0, 0.6))
    _arrow(ax, (90, 31), (66, 54), "playbook (v3)", color=c3, ls="--", toff=(2.5, -1.0), ha="left")

    ax.text(1, 1.5, "Component added in:", fontsize=6.8, va="center")
    for i, (lab, col) in enumerate([("v1", c1), ("v2", c2), ("v3", c3)]):
        ax.plot([24 + i * 7], [1.5], marker="s", ms=4, color=col, lw=0, clip_on=False)
        ax.text(25.5 + i * 7, 1.5, lab, fontsize=6.8, va="center")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"coach_architecture.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 8: lineage + forest
def fig8(data: dict[str, tuple[list[dict], list[dict]]]) -> None:
    plt.rcParams.update(STYLE)
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(W * 1.2, 4.4), gridspec_kw={"height_ratios": [0.85, 1.45], "hspace": 0.3}
    )
    # --- (a) lineage (ordinal spacing; dates only for measured versions)
    n_l = len(LINEAGE)
    ax.set_xlim(-0.8, n_l - 0.2)
    ax.set_ylim(-1.7, 2.35)
    ax.axis("off")
    ax.plot([-0.4, n_l - 0.6], [0, 0], color="#bbbbbb", lw=1.0, zorder=1)
    vcol = {v["key"]: v["color"] for v in VERSIONS}
    dates = {"v3": "Sep 5", "v5": "Sep 8", "v7p": "Sep 9"}
    for i, (name, _x, thesis, kind) in enumerate(LINEAGE):
        if kind in ("rejected", "not run"):
            col = "#b2182b" if kind == "rejected" else "#999999"
            ax.plot([i, i], [0, -0.7], color=col, lw=0.6, zorder=2)
            if kind == "rejected":
                ax.plot(i, -0.7, marker="x", ms=5, color=col, lw=0, zorder=3)
            else:
                ax.plot(i, -0.7, marker="o", mfc="white", mec=col, ms=4, lw=0, zorder=3)
            ax.text(i, -0.95, name, ha="center", va="top", fontsize=6.2, color=col)
        else:
            col = vcol[thesis]
            ms = 7 if kind == "measured" else 4.2
            ax.plot(i, 0, marker="o", ms=ms, color=col, mec="white", mew=0.6, lw=0, zorder=3)
            ax.text(i, 0.28, name, ha="center", va="bottom", fontsize=6.4, color=col)
            if name in dates:
                ax.text(i, -0.3, dates[name], ha="center", va="top", fontsize=5.8, color="#666666")
    idx = {name: i for i, (name, *_r) in enumerate(LINEAGE)}
    brackets = [("v1", "v1", "v3"), ("v2", "v4", "v5"), ("v3", "v5p", "v7p")]
    for key, a, b in brackets:
        x0, x1 = idx[a] - 0.3, idx[b] + 0.3
        ax.plot([x0, x0, x1, x1], [0.95, 1.15, 1.15, 0.95], color=vcol[key], lw=1.0)
        ax.text(
            (x0 + x1) / 2,
            1.28,
            f"thesis {key}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=vcol[key],
            weight="bold",
        )
    ax.text(
        (idx["v8"] + idx["v13"]) / 2,
        -1.45,
        "automated candidates (meta-LLM), rejected",
        ha="center",
        va="top",
        fontsize=6.2,
        color="#b2182b",
    )
    ax.text(
        idx["v13"] + 0.45, -0.7, "not run", ha="left", va="center", fontsize=5.8, color="#999999"
    )
    ax.text(
        -0.8,
        2.35,
        "(a) Internal iterations and the three thesis versions",
        fontsize=8,
        weight="bold",
        va="top",
    )

    # --- (b) forest plot
    rows = []
    for v in VERSIONS:
        ctrl, coach = data[v["key"]]
        p = paired(ctrl, coach, "objective")
        rows.append(
            (f"thesis {v['key']} vs PPO  (n = {p['n']})", p["delta"], p["ci95"], v["color"], "o")
        )
    for _, r in _candidates().iterrows():
        defect = "driver defect" in str(r["note"])
        rows.append(
            (
                f"{r['version']} vs incumbent  (n = {int(r['n_pairs'])})",
                r["dJ_mean"],
                r["ci95_halfwidth"],
                "#b2182b",
                "s" if not defect else "D",
            )
        )
    y = np.arange(len(rows))[::-1]
    for yi, (_lab, d, ci, col, mk) in zip(y, rows, strict=True):
        bx.errorbar(
            d,
            yi,
            xerr=ci,
            fmt=mk,
            color=col,
            ms=4.5,
            capsize=2.5,
            lw=1.0,
            mfc=col if mk != "D" else "white",
        )
    bx.axvline(0, color="#444444", lw=0.8)
    bx.axhspan(y[3] + 0.5, y[0] + 0.6, color="#f0f0f0", zorder=0)
    bx.set_yticks(y)
    bx.set_yticklabels([r[0] for r in rows], fontsize=7)
    bx.set_xlabel("Paired ΔJ (mean, 95 % CI)")
    bx.set_xlim(-1.0, 1.4)
    bx.text(-0.98, y[0] + 0.15, "measured versions", fontsize=6.4, color="#555555")
    bx.text(0.62, y[3] - 0.05, "meta-LLM candidates (all rejected)", fontsize=6.4, color="#b2182b")
    bx.text(0.62, y[-1] - 0.05, "open marker: driver-defect batch", fontsize=6.2, color="#b2182b")
    bx.set_title("(b) Paired objective change", loc="left", fontsize=8, weight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"coach_version_lineage.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 9: J curves 3x3
def _mean_curve(runs: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    grid = np.arange(1, 41) * 1e6
    ys = []
    for r in runs:
        ys.append(np.interp(grid, r["_steps"], r[key], left=np.nan, right=r[key][-1]))
    return grid / 1e6, np.nanmean(np.array(ys), axis=0)


def fig9(data: dict[str, tuple[list[dict], list[dict]]]) -> None:
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(3, 3, figsize=(W * 1.3, 5.6), sharex=True, sharey=True)
    for i, v in enumerate(VERSIONS):
        ctrl, coach = data[v["key"]]
        for j, s in enumerate(SETTINGS):
            ax = axes[i, j]
            c = [r for r in ctrl if r["setting"] == s]
            k = [r for r in coach if r["setting"] == s]
            for r in c:
                ax.plot(r["_steps"] / 1e6, r["_J"], color=PPO_COLOR, lw=0.6, alpha=SEED_ALPHA)
            for r in k:
                ax.plot(r["_steps"] / 1e6, r["_J"], color=v["color"], lw=0.6, alpha=SEED_ALPHA)
            x, m = _mean_curve(c, "_J")
            ax.plot(x, m, color=PPO_COLOR, lw=1.6, ls="--")
            x, m = _mean_curve(k, "_J")
            ax.plot(x, m, color=v["color"], lw=1.6)
            ax.text(
                0.03,
                0.96,
                f"n = {len(k)} + {len(c)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.2,
                color="#555555",
            )
            if i == 0:
                ax.set_title(SLABEL[s], fontsize=8.5)
            if j == 0:
                ax.set_ylabel("Objective J")
                ax.text(
                    -0.36,
                    0.5,
                    f"thesis {v['key']}",
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=v["color"],
                    weight="bold",
                )
            ax.set_ylim(-0.05, 1.85)
    for ax in axes[-1]:
        ax.set_xlabel("Environment steps (M)")
    handles = [Line2D([], [], color=PPO_COLOR, lw=1.6, ls="--", label="PPO (mean)")]
    handles += [
        Line2D([], [], color=v["color"], lw=1.6, label=f"LLM-PPO thesis {v['key']} (mean)")
        for v in VERSIONS
    ]
    handles.append(Line2D([], [], color="#999999", lw=0.6, label="individual seeds"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.01),
        handlelength=1.8,
    )
    fig.subplots_adjust(hspace=0.12, wspace=0.08, bottom=0.12)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"coach_objective_by_version.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 10: curriculum lever
def fig10(data: dict[str, tuple[list[dict], list[dict]]]) -> None:
    plt.rcParams.update(STYLE)
    s = "stairs-medium-traditional"
    fig, axes = plt.subplots(1, 3, figsize=(W * 1.3, 2.5), sharey=True)
    for ax, v in zip(axes, VERSIONS, strict=True):
        ctrl, coach = data[v["key"]]
        k = sorted([r for r in coach if r["setting"] == s], key=lambda r: r["seed"])
        shades = _shades(v["color"], len(k))
        for r, col in zip(k, shades, strict=True):
            steps, lever = r["_lever"]
            if not len(steps):
                continue
            x = np.concatenate([[0], steps]) / 1e6
            y = np.concatenate([[1.0], lever])
            ax.step(x, y, where="post", color=col, lw=0.9)
            ev = r["_events"]
            for st in ev["restore"]:
                ax.plot(
                    st / 1e6,
                    np.interp(st, steps, lever),
                    marker="^",
                    ms=4,
                    color=col,
                    lw=0,
                    mec="black",
                    mew=0.4,
                )
            for st in ev["rollback"]:
                ax.plot(st / 1e6, np.interp(st, steps, lever), marker="x", ms=4, color=col, lw=0)
            onset = r["first_success50_Msteps"]
            if not math.isnan(onset):
                ax.plot(
                    onset,
                    np.interp(onset * 1e6, steps, lever),
                    marker="o",
                    ms=4.5,
                    mfc="white",
                    mec=col,
                    mew=1.0,
                    lw=0,
                )
        ax.axhline(1.0, color="#999999", lw=0.6, ls=":")
        ax.set_title(f"thesis {v['key']}", fontsize=8.5, color=v["color"], weight="bold")
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylim(0.25, 1.6)
        n_restore = sum(len(r["_events"]["restore"]) for r in k)
        n_roll = sum(len(r["_events"]["rollback"]) for r in k)
        ax.text(
            0.03,
            0.04,
            f"n = {len(k)} runs\n{n_restore} restores, {n_roll} rollbacks",
            transform=ax.transAxes,
            fontsize=6.2,
            color="#444444",
        )
    axes[0].set_ylabel("Commanded forward speed (m/s)")
    handles = [
        Line2D(
            [],
            [],
            marker="^",
            ms=4,
            color="#777777",
            mec="black",
            mew=0.4,
            lw=0,
            label="best-snapshot restore",
        ),
        Line2D([], [], marker="x", ms=4, color="#777777", lw=0, label="rollback"),
        Line2D(
            [],
            [],
            marker="o",
            ms=4.5,
            mfc="white",
            mec="#777777",
            mew=1.0,
            lw=0,
            label="success rate ≥ 0.5 first reached",
        ),
        Line2D([], [], color="#999999", lw=0.6, ls=":", label="initial command (1.0 m/s)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=6.8,
        bbox_to_anchor=(0.5, -0.12),
        handlelength=1.6,
    )
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"coach_curriculum_lever.{ext}", bbox_inches="tight")
    plt.close(fig)


def _shades(hex_color: str, n: int) -> list[tuple]:
    base = np.array(matplotlib.colors.to_rgb(hex_color))
    if n <= 1:
        return [tuple(base)]
    out = []
    for i in range(n):
        t = 0.55 * i / max(1, n - 1)  # mix towards white for later seeds
        out.append(tuple(base * (1 - t) + np.array([1, 1, 1]) * t))
    return out


# ------------------------------------------------------------------ fig 11: naive recovery
def fig11(data: dict[str, tuple[list[dict], list[dict]]]) -> None:
    plt.rcParams.update(STYLE)
    s = "rough-medium-naive"
    fig, axes = plt.subplots(1, 3, figsize=(W * 1.3, 2.5), sharey=True)
    for ax, v in zip(axes, VERSIONS, strict=True):
        ctrl, coach = data[v["key"]]
        c = [r for r in ctrl if r["setting"] == s]
        k = sorted([r for r in coach if r["setting"] == s], key=lambda r: r["seed"])
        for r in c:
            ax.plot(r["_steps"] / 1e6, r["_succ"], color=PPO_COLOR, lw=0.7, ls="--", alpha=0.7)
        shades = _shades(v["color"], len(k))
        for r, col in zip(k, shades, strict=True):
            ax.plot(r["_steps"] / 1e6, r["_succ"], color=col, lw=0.9)
        esc = sum(r["escaped"] for r in k)
        esc_c = sum(r["escaped"] for r in c)
        ax.axhline(FLOOR, color="#999999", lw=0.6, ls=":")
        ax.set_title(f"thesis {v['key']}", fontsize=8.5, color=v["color"], weight="bold")
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylim(-0.03, 1.05)
        ax.text(
            0.96,
            0.40,
            f"LLM-PPO escaped {esc}/{len(k)}\nPPO escaped {esc_c}/{len(c)}",
            transform=ax.transAxes,
            fontsize=6.6,
            va="center",
            ha="right",
            color="#333333",
        )
    axes[0].set_ylabel("Success rate (naive reward)")
    handles = [
        Line2D([], [], color=VERSIONS[1]["color"], lw=0.9, label="LLM-PPO seeds"),
        Line2D([], [], color=PPO_COLOR, lw=0.7, ls="--", label="PPO seeds"),
        Line2D([], [], color="#999999", lw=0.6, ls=":", label=f"escape floor ({FLOOR})"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.8,
        bbox_to_anchor=(0.5, -0.12),
        handlelength=1.8,
    )
    fig.subplots_adjust(wspace=0.08)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"coach_naive_recovery.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ fig 12: decision activity
def fig12(data: dict[str, tuple[list[dict], list[dict]]]) -> None:
    plt.rcParams.update(STYLE)
    cats = [
        ("kept", "kept", "#9ecae1"),
        ("rolled_back", "rolled back", "#e08214"),
        ("restores", "best-snapshot restore", "#b2182b"),
        ("refused", "refused by layer", "#4d4d4d"),
    ]
    # 'kept' counts settled interventions; clipped proposals are still applied (so counted in
    # kept/rolled back) and are overlaid as a hatched share of the bar.
    fig, ax = plt.subplots(figsize=(W * 1.2, 2.9))
    x, labels, group_x = [], [], []
    pos = 0
    for v in VERSIONS:
        ctrl, coach = data[v["key"]]
        start = pos
        for s in SETTINGS:
            k = [r for r in coach if r["setting"] == s]
            st = pd.DataFrame([per_run_stats(r) for r in k])
            bottom = 0.0
            for key, lab, col in cats:
                val = st[key].mean()
                ax.bar(
                    pos,
                    val,
                    bottom=bottom,
                    color=col,
                    width=0.72,
                    edgecolor="white",
                    lw=0.4,
                    label=lab if pos == 0 else None,
                )
                bottom += val
                if key == "rolled_back":
                    n_int = bottom
            clip = st["clipped"].mean()
            ax.bar(
                pos,
                clip,
                color="none",
                width=0.72,
                edgecolor="#333333",
                lw=0.4,
                hatch="////",
                label="guardrail-clipped proposal" if pos == 0 else None,
            )
            ax.text(
                pos,
                bottom + 0.25,
                f"{n_int:.1f}",
                ha="center",
                va="bottom",
                fontsize=6,
                color="#444444",
            )
            x.append(pos)
            short = SLABEL[s].split(" ")[0].replace(",", "")
            labels.append(short + ("\n(naive)" if "naive" in s else ""))
            pos += 1
        group_x.append(((start + pos - 1) / 2, v))
        pos += 0.7
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.set_ylim(0, 23)
    for gx, v in group_x:
        ax.text(
            gx,
            22.7,
            f"thesis {v['key']}",
            ha="center",
            va="top",
            fontsize=8,
            color=v["color"],
            weight="bold",
        )
    ax.set_ylabel("Decision events per run (mean)")
    ax.text(
        0.01,
        0.90,
        "numbers above bars: settled interventions per run",
        transform=ax.transAxes,
        fontsize=6.2,
        color="#555555",
        va="top",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=5,
        frameon=False,
        fontsize=6.4,
        handlelength=1.4,
        columnspacing=1.2,
    )
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"coach_decision_activity.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    data = {v["key"]: load_version(v) for v in VERSIONS}
    for v in VERSIONS:
        ctrl, coach = data[v["key"]]
        p = paired(ctrl, coach, "objective")
        print(
            f"{v['key']}: n={p['n']} dJ={p['delta']:+.3f} ci={p['ci95']:.3f} p_t={p['p_t']:.3f} "
            f"p_w={p['p_w']:.3f} d={p['d']:.2f}"
        )
    t9 = table9(data).set_index("Version").T.reset_index().rename(columns={"index": "Item"})
    t9.to_csv(TAB / "coach_versions_thesis.csv", index=False)
    table11(data).to_csv(TAB / "coach_intervention_stats.csv", index=False)
    table_recipe().to_csv(TAB / "coach_naive_recipe.csv", index=False)
    fig7()
    fig8(data)
    fig9(data)
    fig10(data)
    fig11(data)
    fig12(data)
    print("done")


if __name__ == "__main__":
    main()
