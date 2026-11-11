"""Static versions of the paper figures (PNG and SVG) in outputs/figures/, drawn with matplotlib from the
saved outputs. The interactive versions are in the notebooks and on the page.

    python -m analysis.scripts_text.paper_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import s05_scatter, s06_null, s09_logreg, s10_franchise
from .config import cfg
from .paths import OUTPUTS
from .report import HIGH, LOW, NEUTRAL

FIG_DIR = OUTPUTS / "figures"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "scripts-text", "figure.dpi": 150})


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def fig_null():
    r = s06_null.load()
    s, cnt = r["summary"], r["counts"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(cnt.terms_above_threshold, bins=35, color=NEUTRAL, label="1,000 stratified shuffles")
    ax.axvline(s["count_test"]["real"], color=LOW, lw=2.5, ls="-",
               label=f"real: {s['count_test']['real']} (z = {s['count_test']['z']:.2f}, p = {s['count_test']['p']:.3f})")
    ax.set_xlabel("terms at least twice as frequent in one group")
    ax.set_ylabel("shuffles")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Primary comparison against its null", loc="left")
    _save(fig, "fig1_null_count")


def fig_franchise():
    r = s10_franchise.load()
    res, rd = r["results"], r["random_drop"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axhspan(-1.96, 1.96, color=NEUTRAL, alpha=0.2, lw=0)
    labels = ["all films", "without franchise films", "without sequels", "without possible transcripts"]
    xs = np.arange(len(res))
    ax.plot(xs, res.z, "o", color=HIGH, ms=8, label="real comparison")
    jitter = np.random.default_rng(cfg()["seed"]).uniform(-0.12, 0.12, len(rd))
    ax.plot(len(res) + jitter, rd.z, "v", color=LOW, ms=4, alpha=0.7,
            label=f"control: {len(rd)} random drops of the same size")
    ax.set_xticks(list(xs) + [len(res)], labels + ["random drops\n(control)"], rotation=15, ha="right")
    ax.set_ylabel("z against a 200-shuffle stratified null")
    ax.text(len(res) + 0.45, -1.85, "grey band: |z| < 1.96", color="#555", fontsize=7.5, ha="right")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_title("The excess goes with the franchise films", loc="left")
    _save(fig, "fig2_franchise_check")


def fig_auc():
    r = s09_logreg.load()
    cv, perm = r["cv"], r["perm"]
    data = [cv.auc_baseline, cv.auc_text, cv.auc_text_plus_baseline, perm.auc_text]
    labels = ["genre + decade\n+ budget", "text\n(TF-IDF)", "text +\nbaseline", "text,\nshuffled labels"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for patch, colour, hatch in zip(bp["boxes"], [NEUTRAL, HIGH, "#009E73", LOW], ["", "//", "..", "xx"]):
        patch.set_facecolor(colour)
        patch.set_alpha(0.6)
        patch.set_hatch(hatch)
    ax.axhline(0.5, color="#444", ls="--", lw=1)
    ax.text(4.45, 0.505, "chance", color="#444", fontsize=7.5, ha="right")
    ax.set_ylabel("cross-validated AUC (test folds)")
    ax.set_title("Dialogue adds nothing clear to genre, decade and budget", loc="left")
    _save(fig, "fig3_auc")


def fig_scatter():
    t = s05_scatter.load()["primary"]
    e = cfg()["scattertext"]["smoothing_per_10k"]
    thr = cfg()["scattertext"]["threshold"]
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    mid = t.score.abs() < thr
    ax.scatter(t.rate_low[mid] + e, t.rate_high[mid] + e, s=2, color=NEUTRAL, alpha=0.3, lw=0,
               label="less than 2× in either group")
    hi, lo = t.score >= thr, t.score <= -thr
    ax.scatter(t.rate_low[hi] + e, t.rate_high[hi] + e, s=10, marker="o", color=HIGH, label="≥ 2× in successful")
    ax.scatter(t.rate_low[lo] + e, t.rate_high[lo] + e, s=12, marker="v", color=LOW, label="≥ 2× in unsuccessful")
    lim = [min(t.rate_low.min(), t.rate_high.min()) + e, max(t.rate_low.max(), t.rate_high.max()) + e]
    ax.plot(lim, lim, ls="--", color="#444", lw=1)
    for _, r_ in list(t.head(8).iterrows()) + list(t.tail(8).iterrows()):
        ax.annotate(r_.term, (r_.rate_low + e, r_.rate_high + e), fontsize=6.5, xytext=(2, 2), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"unsuccessful scripts, per 10,000 words (+{e})")
    ax.set_ylabel(f"successful scripts, per 10,000 words (+{e})")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Primary Scattertext (supplement)", loc="left")
    _save(fig, "figS1_scattertext")


def main():
    for f in (fig_null, fig_franchise, fig_auc, fig_scatter):
        f()
    print(f"wrote {sorted(p.name for p in FIG_DIR.iterdir())}")


if __name__ == "__main__":
    main()
