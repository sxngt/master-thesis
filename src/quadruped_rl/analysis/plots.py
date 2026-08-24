"""Publication-quality figures (matplotlib/seaborn). All thesis figures are
generated here for consistency; save as PDF (vector) + PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {
    "figure.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "savefig.bbox": "tight",
}


def _save(fig, out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


def learning_curves(
    curves: dict[str, list[tuple[np.ndarray, np.ndarray]]], metric_label: str, out: str | Path
) -> None:
    """curves: {algorithm: [(steps, values) per seed]} -> mean ± std band."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for name, seeds in curves.items():
            grid = seeds[0][0]
            stacked = np.stack([np.interp(grid, s, v) for s, v in seeds])
            mean, std = stacked.mean(0), stacked.std(0)
            ax.plot(grid, mean, label=name.upper())
            ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(metric_label)
        ax.legend(fontsize=7)
        _save(fig, out)


def metric_heatmap(results: pd.DataFrame, metric: str, out: str | Path) -> None:
    """Algorithm x terrain heatmap of a metric (mean over robots and seeds)."""
    import seaborn as sns

    pivot = results.pivot_table(index="algorithm", columns="terrain", values=metric, aggfunc="mean")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", ax=ax)
        ax.set_title(metric)
        _save(fig, out)


def comparison_box(results: pd.DataFrame, metric: str, out: str | Path) -> None:
    """Per-algorithm distribution boxplot across seeds."""
    import seaborn as sns

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sns.boxplot(data=results, x="algorithm", y=metric, ax=ax)
        sns.stripplot(
            data=results, x="algorithm", y=metric, ax=ax, color="black", size=2, alpha=0.5
        )
        _save(fig, out)
