"""Figures for the paper, drawn only from saved, committed outputs (no raw data, no script text).

    python paper/make_figures.py        # from the repo root; writes paper/figures/*.pdf

Two inputs are not yet files on main and are copied here with their source:
  - the out-of-time R2 per block, printed by notebooks/what_predicts_profit.ipynb (Summary cell);
  - the money-vs-liking correlations, from analysis/reviews/README.md on branch reviews-exploration.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Hard-coded input. tests/test_paper_numbers.py checks it against its source, so a rerun that moves it fails.
# (The out-of-time ridge R2 ladder that was Fig. 1A is gone: ridge isn't a course method.)
# Fig. 3: analysis/reviews/outputs/spearman_correlations.csv (branch reviews-exploration): Spearman rho with
# ln(gross/budget) on films to 2017, and the mean and SD of a null that shuffles within genre x decade.
RECEPTION = [("IMDb rating", 0.266, 0.099, 0.018), ("Metacritic", 0.261, 0.089, 0.020),
             ("Rotten Tomatoes", 0.326, 0.122, 0.018),
             ("Review tone (VADER)", 0.127, 0.025, 0.017)]  # review tone from that branch's README; SD from z = 6.10

INK, ACCENT, GREY, LIGHT = "#1F2933", "#1F6F8B", "#8A969C", "#D5E3E8"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "pdf.fonttype": 42})


def panel(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def fig_network():
    # Return similarity by role, adjusted for ln budget, decade and genre (analysis/network/roi_similarity.json).
    r = json.loads((ROOT / "src/analysis/network/roi_similarity.json").read_text())["results"]
    adj, ws = r["adjusted"]["roles"], r["adjusted_within_studio"]["roles"]
    roles = sorted(adj, key=lambda k: adj[k]["similarity"])
    fig, b = plt.subplots(figsize=(4.0, 2.3))
    y = np.arange(len(roles))
    b.barh(y, [adj[k]["similarity"] for k in roles], color=ACCENT, height=0.6, label="Observed")
    b.scatter([adj[k]["similarity_null_95"] for k in roles], y, marker="|", s=60, color=INK, zorder=3,
              label="Shuffle null, 95th pct.")
    wy = [i for i, k in enumerate(roles) if k in ws]
    b.scatter([ws[roles[i]]["similarity_null_95"] for i in wy], wy, marker="x", s=14, color="#B5482F", zorder=3,
              label="Null within studio, 95th pct.")
    b.set_yticks(y, [k.capitalize() for k in roles])
    b.set_xlabel("Return similarity of films sharing the role (ICC-like)")
    b.set_title("Films sharing a person or company have similar returns")
    b.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3)
    fig.savefig(OUT / "fig_network.pdf", bbox_inches="tight")


def fig_script():
    s10 = pd.read_csv(ROOT / "archive/analysis/scripts_text/outputs/s10_results.csv").set_index("run")
    drops = pd.read_csv(ROOT / "archive/analysis/scripts_text/outputs/s10_random_drop.csv")
    fig, a = plt.subplots(figsize=(3.45, 2.2))
    a.scatter(np.random.default_rng(1).uniform(-0.12, 0.12, len(drops)) + 1, drops.z, s=5, color=GREY,
              label="50 random drops")
    a.scatter([0], [s10.loc["primary", "z"]], s=22, color=ACCENT, zorder=3)
    a.scatter([2], [s10.loc["without likely franchise films", "z"]], s=22, color="#B5482F", zorder=3)
    a.axhline(1.96, color=INK, lw=0.5, ls="--")
    a.set_xticks([0, 1, 2], ["All", "Random\ndrop", "Franch.\ndrop"], fontsize=6)
    a.set_ylabel("Word difference vs shuffle (z)")
    a.set_xlim(-0.5, 2.5)
    fig.savefig(OUT / "fig_script.pdf", bbox_inches="tight")


def fig_reception():
    rows = RECEPTION
    fig, ax = plt.subplots(figsize=(3.45, 1.7))
    y = np.arange(len(rows))[::-1]
    for yi, (k, rho, m, sd) in zip(y, rows):
        ax.plot([m - 1.96 * sd, m + 1.96 * sd], [yi, yi], color=GREY, lw=5, solid_capstyle="butt")
        ax.scatter([rho], [yi], color=ACCENT, s=22, zorder=3)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_xlabel("Spearman $\\rho$ with ln(gross/budget)")
    ax.set_title("Dots: observed. Grey bars: 95% of a genre × decade shuffle null", fontsize=6.5, color=INK)
    fig.savefig(OUT / "fig_reception.pdf", bbox_inches="tight")


def fig_head_to_head():
    # Return assortativity of the three networks on the common set (analysis/compare/head_to_head.py).
    d = pd.read_csv(ROOT / "src/analysis/compare/outputs/head_to_head/assortativity.csv").set_index(["network", "links"])
    rows = [("people", "all links", "Shared people: all links"), ("people", "links within a shared genre", "same genre"),
            ("people", "links with no shared genre", "different genres"),
            ("script", "all links", "Similar screenplays: all links"), ("script", "links within a shared genre", "same genre"),
            ("script", "links with no shared genre", "different genres"), ("genre", "all links", "Shared genre: all links")]
    y = np.array([0, 1, 2, 3.5, 4.5, 5.5, 7.0])
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    v = [d.loc[r[:2], "assortativity"] for r in rows]
    ax.barh(y, v, height=0.7, color=[ACCENT if r[1] == "all links" else "#9CC3D1" for r in rows])
    ax.scatter([d.loc[r[:2], "null_95"] for r in rows], y, marker="|", s=70, color="#B5482F", zorder=3,
               label="95th percentile of 1,000 shuffles")
    for yi, x, r in zip(y, v, rows):
        n95 = d.loc[r[:2], "null_95"]
        ax.text(0.0015 if x < 0 else max(x, n95) + 0.0015, yi, f"{x:.4f}" if abs(x) < 0.001 else f"{x:.3f}", va="center", fontsize=6)
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(y, [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 0.06)
    ax.set_xlabel("Return assortativity")
    ax.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.1)
    fig.savefig(OUT / "fig_head_to_head.pdf", bbox_inches="tight")


if __name__ == "__main__":
    fig_head_to_head()
    fig_network()
    fig_script()
    fig_reception()
    print("written:", sorted(p.name for p in OUT.glob("*.pdf")))
