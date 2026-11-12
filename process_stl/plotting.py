"""Matplotlib / seaborn box plots of the STL statistics (see stats.py)."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from .stats import AXES, TEST_NAMES, format_p, metrics

#: which metrics to plot -> figure title
PLOT_KINDS = {
    "all": "Center of mass (top row) and bounding box size (bottom row)",
    "com": "Center of mass",
    "size": "Bounding box size",
}


def annotate_pvalues(ax, df, col, order, pvals):
    """Draw a bracket with the p-value above each pair of boxes."""
    lo, hi = df[col].min(), df[col].max()
    if not np.isfinite(hi):
        return
    span = (hi - lo) or 1.0
    y = hi + 0.05 * span

    # Adjacent pairs first so brackets stack neatly when there are >2 folders
    pairs = sorted(combinations(range(len(order)), 2), key=lambda ij: ij[1] - ij[0])
    for i, j in pairs:
        p = pvals.get((col, order[i], order[j]), np.nan)
        tick = 0.02 * span
        ax.plot([i, i, j, j], [y, y + tick, y + tick, y], color="black", lw=1)
        ax.text((i + j) / 2, y + tick, format_p(p), ha="center", va="bottom", fontsize=9)
        y += 0.14 * span

    ax.set_ylim(top=y)


def plot_boxplots(df, order, pvals_df, test, kind="all", axes=AXES,
                  display_labels=None, unit="mm", fig=None) -> Figure:
    """One box plot (+ individual files) per metric, with p-value brackets.

    kind: "all" (2x3 grid), "com" or "size" (1x3).
    display_labels: label -> text shown on the x axis (default: the label).
    fig: draw into this figure (e.g. a Qt canvas) instead of a new one.
    """
    if kind not in PLOT_KINDS:
        raise ValueError(f"unknown kind {kind!r}, choose from {list(PLOT_KINDS)}")
    cols = {"all": metrics(axes), "com": metrics(axes)[:3], "size": metrics(axes)[3:]}[kind]
    nrows = 2 if kind == "all" else 1

    sns.set_theme(style="whitegrid")
    if fig is None:
        fig = plt.figure(figsize=(13, 8) if nrows == 2 else (11, 4))
    fig.clear()
    axs = fig.subplots(nrows, 3, squeeze=False)
    pvals = pvals_df.set_index(["metric", "group1", "group2"])["p_value"].to_dict()
    tick_labels = [(display_labels or {}).get(lab, lab) for lab in order]

    for ax, col in zip(axs.flat, cols):
        sns.boxplot(data=df, x="label", y=col, hue="label", order=order,
                    hue_order=order, legend=False, showfliers=False, ax=ax)
        # Individual files on top (outliers are shown here, not by the boxplot)
        sns.stripplot(data=df, x="label", y=col, order=order, color="black",
                      size=3, alpha=0.6, jitter=0.2, ax=ax)
        annotate_pvalues(ax, df, col, order, pvals)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks(range(len(order)), tick_labels)
    for row in axs:
        row[0].set_ylabel(unit)

    fig.suptitle(f"{PLOT_KINDS[kind]}\np-values: {TEST_NAMES[test]}")
    fig.tight_layout()
    return fig
