"""Black-and-white-safe plotting helpers with labelled axes."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Print-safe greyscale markers / lines.
MARKERS = ("o", "s", "^", "D", "v", "P", "X")
LINESTYLES = ("-", "--", "-.", ":")
GREYS = ("#111111", "#444444", "#777777", "#AAAAAA")


def apply_bw_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "grid.color": "#CCCCCC",
            "grid.linestyle": ":",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.edgecolor": "black",
        }
    )


def scatter_bw(
    ax,
    x,
    y,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    alpha: float = 0.35,
    marker: str = "o",
) -> None:
    ax.scatter(x, y, c="black", alpha=alpha, marker=marker, edgecolors="none", s=18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both")


def save_fig(fig, path: Path, caption: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    if caption:
        path.with_suffix(".caption.txt").write_text(caption.strip() + "\n", encoding="utf8")
    plt.close(fig)
    return path


def quadrant_mask(x: pd.Series, y: pd.Series, x_thr: float, y_thr: float) -> dict[str, pd.Series]:
    """Four quadrants relative to thresholds (e.g. median or break-even)."""
    return {
        "high_x_high_y": (x >= x_thr) & (y >= y_thr),
        "high_x_low_y": (x >= x_thr) & (y < y_thr),
        "low_x_high_y": (x < x_thr) & (y >= y_thr),
        "low_x_low_y": (x < x_thr) & (y < y_thr),
    }


def annotate_extremes(ax, df: pd.DataFrame, xcol: str, ycol: str, labelcol: str, n: int = 5) -> None:
    """Label a few extreme points for readability (disagreement movies)."""
    if df.empty:
        return
    # Rank by distance from thresholds already applied upstream; here just top |y| extremes.
    score = (df[xcol] - df[xcol].median()).abs() + (df[ycol] - df[ycol].median()).abs()
    top = df.loc[score.nlargest(n).index]
    for _, row in top.iterrows():
        ax.annotate(
            str(row[labelcol])[:28],
            (row[xcol], row[ycol]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
