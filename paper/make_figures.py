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

INK, ACCENT, GREY, LIGHT = "#1F2933", "#1F6F8B", "#8A969C", "#D5E3E8"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "pdf.fonttype": 42})


def panel(ax, letter):
    ax.text(-0.18, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def fig_network():
    # A: out-of-time R2 (train <= 2010, test 2011-2017), ridge; notebooks/what_predicts_profit.ipynb summary.
    ladder = [("Budget,\nyear,\ngenre", -0.010), ("+ net-\nwork", 0.202), ("+ month,\nfran-\nchise", 0.244),
              ("+ critics,\naudi-\nence*", 0.312)]
    # B: return similarity by role, adjusted for ln budget, decade and genre (analysis/network/roi_similarity.json).
    r = json.loads((ROOT / "analysis/network/roi_similarity.json").read_text())["results"]
    adj, ws = r["adjusted"]["roles"], r["adjusted_within_studio"]["roles"]
    roles = sorted(adj, key=lambda k: adj[k]["similarity"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.45})
    x = np.arange(len(ladder))
    a.bar(x, [v for _, v in ladder], color=[GREY, ACCENT, ACCENT, LIGHT], edgecolor=INK, linewidth=0.4, width=0.65)
    for i, (_, v) in enumerate(ladder):
        a.text(i, v + 0.008 if v > 0 else 0.008, f"{v:.2f}", ha="center", fontsize=6.5)
    a.axhline(0, color=INK, lw=0.5)
    a.set_xticks(x, [k for k, _ in ladder])
    a.set_ylabel("Out-of-sample $R^2$, ln(gross/budget)")
    a.set_title("Films 2011–2017, model fitted on films to 2010")
    panel(a, "A")
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
    panel(b, "B")
    fig.savefig(OUT / "fig_network.pdf", bbox_inches="tight")


def fig_script():
    s10 = pd.read_csv(ROOT / "analysis/scripts_text/outputs/s10_results.csv").set_index("run")
    drops = pd.read_csv(ROOT / "analysis/scripts_text/outputs/s10_random_drop.csv")
    p7 = json.loads((ROOT / "analysis/arcs/outputs/s07_primary.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 2.2), gridspec_kw={"wspace": 0.8})
    a.scatter(np.random.default_rng(1).uniform(-0.12, 0.12, len(drops)) + 1, drops.z, s=5, color=GREY,
              label="50 random drops")
    a.scatter([0], [s10.loc["primary", "z"]], s=22, color=ACCENT, zorder=3)
    a.scatter([2], [s10.loc["without likely franchise films", "z"]], s=22, color="#B5482F", zorder=3)
    a.axhline(1.96, color=INK, lw=0.5, ls="--")
    a.set_xticks([0, 1, 2], ["All", "Random\ndrop", "Franch.\ndrop"], fontsize=6)
    a.set_ylabel("Word difference vs shuffle (z)")
    a.set_xlim(-0.5, 2.5)
    panel(a, "A")
    for i, d in enumerate(("D1", "D2")):
        res = p7["results"][d]
        b.errorbar([i], [res["rho"]], yerr=[[res["rho"] - res["ci_low"]], [res["ci_high"] - res["rho"]]],
                   fmt="o", color=ACCENT, ecolor=ACCENT, ms=4, capsize=2, lw=0.8)
        b.text(i + 0.12, res["rho"], f"p={res['p']:.3f}", fontsize=6, va="center")
    rmin = p7["r_min"]["alpha_0.025"]
    b.axhspan(-rmin, rmin, color=LIGHT, lw=0)
    b.axhline(0, color=INK, lw=0.5)
    b.set_xticks([0, 1], ["Swing\n(D1)", "Ending lift\n(D2)"])
    b.set_xlim(-0.5, 1.9)
    b.text(1.85, -rmin + 0.006, "shaded: below the\n$\\alpha=0.025$ threshold", fontsize=5.5, ha="right", va="bottom", color=GREY)
    b.set_ylabel("Partial Spearman $\\rho$ with ln(gross/budget)")
    panel(b, "B")
    fig.savefig(OUT / "fig_script.pdf", bbox_inches="tight")


def fig_reception():
    # analysis/reviews/README.md (branch reviews-exploration): Spearman rho with ln(gross/budget), 2,095 films to
    # 2017, and the mean and SD of a null that shuffles within genre x decade.
    rows = [("IMDb rating", 0.266, 0.099, 0.018), ("Metacritic", 0.261, 0.089, 0.020),
            ("Rotten Tomatoes", 0.326, 0.122, 0.018), ("Review tone (VADER)", 0.127, 0.025, 0.017)]  # SD for review tone derived from z = 6.10
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


if __name__ == "__main__":
    fig_network()
    fig_script()
    fig_reception()
    print("written:", sorted(p.name for p in OUT.glob("*.pdf")))
