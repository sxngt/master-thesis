"""Table 8: coach ablation — none / random / hillclimb / LLM (v7p) sharing one decision layer.

All four conditions come from the same batch (evolve4 control + v7p, ablation_v7p random +
hillclimb), same 3 settings x 3 seeds, 40M steps. Every comparison is paired on (setting, seed).

    uv run python paper/make_ablation_table.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from make_coach_figures import SETTINGS, SLABEL, TAB, fmt, load_runs, paired

ROOT = Path(__file__).resolve().parents[1]
ARMS = [  # (label, root)
    ("PPO", "data/results/evolve4/control-none/runs"),
    ("Random", "data/results/ablation_v7p/random/runs"),
    ("Hill-climb", "data/results/ablation_v7p/hillclimb/runs"),
    ("LLM (v7p)", "data/results/evolve4/v7p/runs"),
]


def per_setting(arms: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for s in SETTINGS + ["pooled"]:
        row: dict[str, object] = {"Setting": SLABEL.get(s, "Pooled")}
        for label, runs in arms.items():
            sel = [r for r in runs if s == "pooled" or r["setting"] == s]
            j = np.array([r["objective"] for r in sel])
            row["n"] = len(j)
            row[label] = f"{fmt(j.mean())} ± {fmt(j.std(ddof=1))}" if len(j) > 1 else fmt(j.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts(arms: dict[str, list[dict]]) -> pd.DataFrame:
    pairs = [
        ("Random", "PPO"),
        ("Hill-climb", "PPO"),
        ("LLM (v7p)", "PPO"),
        ("LLM (v7p)", "Random"),
        ("LLM (v7p)", "Hill-climb"),
    ]
    rows = []
    for a, b in pairs:
        for s in SETTINGS + ["no-naive", "pooled"]:
            ref = [r for r in arms[b] if _in(r, s)]
            trt = [r for r in arms[a] if _in(r, s)]
            p = paired(ref, trt, "objective")
            rows.append(
                {
                    "Contrast": f"{a} − {b}",
                    "Setting": SLABEL.get(s, "Rough + Stairs" if s == "no-naive" else "Pooled"),
                    "n pairs": p["n"],
                    "ΔJ": fmt(p["delta"]),
                    "SD": fmt(_sd(ref, trt)) if p["n"] > 1 else "",
                    "95% CI": f"± {fmt(p['ci95'])}" if not math.isnan(p["ci95"]) else "",
                    "p (paired t)": fmt(p["p_t"]),
                    "p (Wilcoxon)": fmt(p["p_w"]),
                    "Cohen's d": fmt(p["d"], 2),
                }
            )
    return pd.DataFrame(rows)


def _in(r: dict, s: str) -> bool:
    if s == "pooled":
        return True
    if s == "no-naive":
        return r["setting"] != "rough-medium-naive"
    return r["setting"] == s


def _sd(ref: list[dict], trt: list[dict]) -> float:
    return float(np.std([t["objective"] - r["objective"] for r, t in _zip(ref, trt)], ddof=1))


def _zip(ref: list[dict], trt: list[dict]) -> list[tuple[dict, dict]]:
    idx = {(r["setting"], r["seed"]): r for r in ref}
    return [(idx[(t["setting"], t["seed"])], t) for t in trt if (t["setting"], t["seed"]) in idx]


def escapes(arms: dict[str, list[dict]]) -> pd.DataFrame:
    """Naive-reward floor escapes per arm, Fisher exact vs the LLM arm."""
    naive = "rough-medium-naive"
    llm = [r for r in arms["LLM (v7p)"] if r["setting"] == naive]
    k_llm = sum(r["escaped"] for r in llm)
    rows = []
    for label, runs in arms.items():
        sel = [r for r in runs if r["setting"] == naive]
        k = sum(r["escaped"] for r in sel)
        table = [[k, len(sel) - k], [k_llm, len(llm) - k_llm]]
        p = math.nan if label == "LLM (v7p)" else stats.fisher_exact(table)[1]
        rows.append(
            {
                "Condition": label,
                "Escaped / n": f"{k} / {len(sel)}",
                "Final J": " / ".join(
                    fmt(r["objective"]) for r in sorted(sel, key=lambda r: r["seed"])
                ),
                "p (Fisher vs LLM)": fmt(p),
            }
        )
    non = [
        r for lab, runs in arms.items() if lab != "LLM (v7p)" for r in runs if r["setting"] == naive
    ]
    k = sum(r["escaped"] for r in non)
    rows.append(
        {
            "Condition": "All LLM-free (PPO + Random + Hill-climb)",
            "Escaped / n": f"{k} / {len(non)}",
            "Final J": "",
            "p (Fisher vs LLM)": fmt(
                stats.fisher_exact([[k, len(non) - k], [k_llm, len(llm) - k_llm]])[1]
            ),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    arms = {label: load_runs(ROOT / root) for label, root in ARMS}
    for label, runs in arms.items():
        assert len(runs) == 9, (label, len(runs))
    per_setting(arms).to_csv(TAB / f"table8_ablation_per_setting{a.suffix}.csv", index=False)
    contrasts(arms).to_csv(TAB / f"table8_ablation_contrasts{a.suffix}.csv", index=False)
    escapes(arms).to_csv(TAB / f"table8_ablation_escapes{a.suffix}.csv", index=False)
    pd.set_option("display.width", 200)
    print(per_setting(arms).to_string(index=False))
    print()
    print(contrasts(arms).to_string(index=False))
    print()
    print(escapes(arms).to_string(index=False))


if __name__ == "__main__":
    main()
