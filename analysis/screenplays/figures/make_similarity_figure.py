"""Figure: hit/flop script separation against its budget-matched shuffle (word2vec/results/budget_matched.csv).
Run from the repo root: python analysis/screenplays/figures/make_similarity_figure.py"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
rows = [r for r in csv.DictReader(open(ROOT / "word2vec/results/budget_matched.csv"))
        if r["shuffle_within"] == "era x budget terciles"]
order = sorted({r["genre"] for r in rows}, key=lambda g: max(float(r["separation"]) for r in rows if r["genre"] == g))
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.6), sharey=True, sharex=True)
for ax, layer in zip(axes, ("dialogue", "description")):
    for k, g in enumerate(order):
        r = next(r for r in rows if r["genre"] == g and r["layer"] == layer)
        ok = float(r["p_holm"]) < .05
        ax.barh(k, float(r["separation"]), color="#1f6fb2" if ok else "#b5b5b5", height=.62)
        ax.plot([float(r["null_q95"])] * 2, [k - .4, k + .4], color="#c0392b", lw=1.4)
    ax.axvline(0, color="k", lw=.6)
    ax.set_title(layer.capitalize(), fontsize=8)
    ax.tick_params(labelsize=6.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_yticks(range(len(order)), [f"{g} ({next(r for r in rows if r['genre'] == g)['hits']}/"
                                       f"{next(r for r in rows if r['genre'] == g)['flops']})" for g in order])
fig.supxlabel("Extra similarity within hits and within flops", fontsize=7)
fig.tight_layout(pad=.5)
for ext in ("pdf", "png"):
    fig.savefig(Path(__file__).parent / f"fig_script_similarity.{ext}", dpi=200)
