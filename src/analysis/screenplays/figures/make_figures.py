"""Draw the candidate figures for the screenplay question (Q3) from the committed result tables only.

Nothing is recomputed: every number comes from analysis/screenplays/word2vec/results/*.csv.

Run from the repo root:  .venv/bin/python analysis/screenplays/figures/make_figures.py
Writes fig1_signal_vs_null.{png,pdf}, fig2_layer_hypothesis.{png,pdf}, fig3_added_value.{png,pdf} here.

Colours: categorical slots 1 and 2 of the validated default palette (blue, orange); everything that is a
reference (null, baseline, votes) is grey or black, so colour always means "which representation".
Markers differ in shape as well, so the figures also read in black and white.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "word2vec" / "results"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
NULL_BAND, REF = "#d9d8d3", "#52514e"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK, "text.color": INK, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})

TESTS = [("Comedy", "dialogue"), ("Comedy", "description"), ("Action", "dialogue"), ("Action", "description")]
YPOS = {("Comedy", "dialogue"): 3.3, ("Comedy", "description"): 2.5,
        ("Action", "dialogue"): 1.1, ("Action", "description"): 0.3}


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def fig1():
    prim = pd.read_csv(RES / "primary.csv")
    sub = pd.read_csv(RES / "subset_100k.csv")
    votes = pd.read_csv(RES / "votes_check.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    panels = [
        ("A  All labelled films", prim, "all", "Comedy 136 hits / 136 flops, Action 108 / 108"),
        ("B  Inside the 100k+ vote tier only", sub, "100k", "Comedy 119 / 42, Action 104 / 45"),
    ]
    for ax, (title, df, kind, note) in zip(axes, panels):
        for genre, layer in TESTS:
            y = YPOS[(genre, layer)]
            null = pd.read_csv(RES / (f"null_{genre}_{layer}.csv" if kind == "all" else f"null_100k_{genre}_{layer}.csv"))["w2v"]
            lo, hi = np.percentile(null, [2.5, 97.5])
            ax.barh(y, hi - lo, left=lo, height=0.55, color=NULL_BAND, zorder=1, linewidth=0)
            w = df[(df.genre == genre) & (df.layer == layer) & (df.representation == "w2v")].iloc[0]
            t = df[(df.genre == genre) & (df.layer == layer) & (df.representation == "tfidf")].iloc[0]
            base = df[(df.genre == genre) & (df.layer == "-")].iloc[0]
            vs = votes[(votes.genre == genre) & (votes.model == "log_votes_raw") &
                       (votes.set == ("all labelled" if kind == "all" else "100k+"))].iloc[0]
            ax.plot(base.auc, y, marker="D", ms=5.5, color=REF, mec=SURFACE, mew=0.8, ls="none", zorder=3)
            ax.plot(vs.auc, y, marker="x", ms=6.5, mew=1.6, color=INK, ls="none", zorder=3)
            ax.plot(t.auc, y, marker="s", ms=6, mfc=SURFACE, mec=ORANGE, mew=1.6, ls="none", zorder=4)
            ax.plot(w.auc, y, marker="o", ms=7, color=BLUE, mec=SURFACE, mew=1.0, ls="none", zorder=5)
            ax.text(1.02, y, f"p = {w.shuffle_p:.3f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7.8, color=INK2)
        ax.axvline(0.5, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
        ax.set_xlim(0.40, 0.97)
        ax.set_ylim(-0.35, 4.45)
        ax.set_xlabel("AUC of hit vs flop (0.5 = chance)")
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=18)
        ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=7.6, color=INK2, va="bottom")
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.text(1.02, 3.95, "Word2Vec\nshuffle p", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7.4, color=INK2, fontweight="bold")
    axes[0].set_yticks([YPOS[t] for t in TESTS])
    axes[0].set_yticklabels([f"{g}, {l}" for g, l in TESTS])
    handles = [
        Line2D([], [], marker="o", color=BLUE, ls="none", ms=7, label="Word2Vec script vector"),
        Line2D([], [], marker="s", mfc=SURFACE, mec=ORANGE, mew=1.6, ls="none", ms=6, label="TF-IDF words"),
        Patch(color=NULL_BAND, label="Shuffled labels (95% of 200)"),
        Line2D([], [], marker="D", color=REF, ls="none", ms=5.5, label="Metadata baseline"),
        Line2D([], [], marker="x", color=INK, mew=1.6, ls="none", ms=6.5, label="log(votes) alone"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, -0.07),
               columnspacing=1.2, handletextpad=0.4)
    fig.subplots_adjust(wspace=0.42, left=0.14, right=0.90, bottom=0.2, top=0.8)
    fig.text(0.0, -0.17, "Metadata baseline: vote tier + era + other genres in A; era + other genres in B (the tier is constant "
             "there).\nGrey band: 2.5th to 97.5th percentile of 200 label shuffles. Word2Vec = mean over 5 seeds.",
             fontsize=7.4, color=INK2, va="top")
    save(fig, "fig1_signal_vs_null")


def fig2():
    c = pd.read_csv(RES / "contrast.csv")
    order = [("Comedy", "w2v"), ("Comedy", "tfidf"), ("Action", "w2v"), ("Action", "tfidf"),
             ("Drama", "w2v"), ("Drama", "tfidf")]
    ypos = {("Comedy", "w2v"): 5.3, ("Comedy", "tfidf"): 4.6, ("Action", "w2v"): 3.0, ("Action", "tfidf"): 2.3,
            ("Drama", "w2v"): 0.7, ("Drama", "tfidf"): 0.0}
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for genre, rep in order:
        r = c[(c.genre == genre) & (c.representation == rep)].iloc[0]
        y = ypos[(genre, rep)]
        colr, mk = (BLUE, "o") if rep == "w2v" else (ORANGE, "s")
        ax.plot([r.ci95_lo, r.ci95_hi], [y, y], color=colr, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(r.dialogue_minus_description, y, marker=mk, ms=7, color=colr if rep == "w2v" else SURFACE,
                mec=colr, mew=1.6, ls="none", zorder=4)
        ax.text(0.185, y, f"{r.dialogue_minus_description:+.3f}", ha="right", va="center", fontsize=7.8, color=INK2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-0.14, 0.19)
    ax.set_ylim(-0.6, 6.3)
    for genre, yy, pred in (("Comedy", 4.95, "predicted: dialogue better (> 0)"),
                            ("Action", 2.65, "predicted: description better (< 0)"),
                            ("Drama", 0.35, "exploratory, no prediction")):
        ax.text(-0.138, yy + 0.95 if genre != "Drama" else yy + 0.95, f"{genre}  |  {pred}", fontsize=8, fontweight="bold",
                color=INK, va="bottom")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUC(dialogue) − AUC(description), with 95% bootstrap interval\n← description better        dialogue better →")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", color=BLUE, ls="none", ms=7, label="Word2Vec"),
               Line2D([], [], marker="s", mfc=SURFACE, mec=ORANGE, mew=1.6, ls="none", ms=6.5, label="TF-IDF")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "fig2_layer_hypothesis")


def fig3():
    comb = pd.read_csv(RES / "combo_interval.csv")
    votes = pd.read_csv(RES / "votes_check.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    for genre, layer in TESTS:
        y = YPOS[(genre, layer)]
        r = comb[(comb.genre == genre) & (comb.layer == layer)].iloc[0]
        v = votes[(votes.genre == genre) & (votes.set == "all labelled")]
        base = v[v.model == "baseline+votes"].auc.iloc[0]
        withv = v[(v.model == "w2v+baseline+votes") & (v.layer == layer)].auc.iloc[0]
        ax.plot([r.diff_ci95_lo, r.diff_ci95_hi], [y + 0.14, y + 0.14], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(r.difference, y + 0.14, marker="o", ms=7, color=BLUE, mec=SURFACE, mew=1, ls="none", zorder=4)
        ax.plot(withv - base, y - 0.14, marker="s", ms=6.5, mfc=SURFACE, mec=ORANGE, mew=1.6, ls="none", zorder=4)
        ax.text(0.19, y + 0.14, f"{r.difference:+.3f}", ha="right", va="center", fontsize=7.8, color=INK2)
        ax.text(0.19, y - 0.14, f"{withv - base:+.3f}", ha="right", va="center", fontsize=7.8, color=INK2)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-0.12, 0.195)
    ax.set_ylim(-0.35, 4.0)
    ax.set_yticks([YPOS[t] for t in TESTS])
    ax.set_yticklabels([f"{g}, {l}" for g, l in TESTS])
    ax.set_xlabel("Change in AUC when the Word2Vec vector is added to the metadata features\n← worse        better →")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker="o", color=BLUE, ls="none", ms=7, label="Added to vote tier + era + genres (95% interval)"),
               Line2D([], [], marker="s", mfc=SURFACE, mec=ORANGE, mew=1.6, ls="none", ms=6.5,
                      label="Added to the same plus log(votes) (no interval)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.28), frameon=False, fontsize=7.8, ncol=1)
    fig.tight_layout()
    save(fig, "fig3_added_value")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
