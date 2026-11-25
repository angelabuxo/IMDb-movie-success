"""Does the 500-token threshold cut through a continuum? (no outcomes, numbers only)

Reads layer_stats.csv (one row per unique id). Writes threshold_table.csv and
threshold_hist.png. The threshold is NOT changed here.
Run: python3 -I analysis/screenplays/data_check/threshold_check.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(HERE / "layer_stats.csv")))
BINS = [(0, 0), (1, 99), (100, 249), (250, 499), (500, 999), (1000, 1999), (2000, 10**9)]
LABEL = ["0", "1-99", "100-249", "250-499", "500-999", "1000-1999", ">=2000"]

out = []
print(f"{'tokens':<11}{'dialogue':>10}{'description':>13}")
tok = {L: np.array([int(r[f"{L}_tokens"]) for r in rows]) for L in ["dialogue", "description"]}
for (lo, hi), lab in zip(BINS, LABEL):
    n = {L: int(((tok[L] >= lo) & (tok[L] <= hi)).sum()) for L in tok}
    print(f"{lab:<11}{n['dialogue']:>10}{n['description']:>13}")
    out.append({"tokens": lab, **n})
with open(HERE / "threshold_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["tokens", "dialogue", "description"])
    w.writeheader()
    w.writerows(out)
for L in tok:
    print(f"\n{L}: the 12 smallest values at or above 100 tokens:", sorted(int(x) for x in tok[L] if x >= 100)[:12])
    print(f"{L}: films between 500 and 2000 tokens: {int(((tok[L] >= 500) & (tok[L] < 2000)).sum())}")
both = int((((tok['dialogue'] >= 500) & (tok['dialogue'] < 2000)) | ((tok['description'] >= 500) & (tok['description'] < 2000))).sum())
print(f"\nfilms between 500 and 2000 in at least one layer: {both}")

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
edges = np.logspace(0, np.log10(50000), 41)
for col, L in enumerate(["dialogue", "description"]):
    x = tok[L]
    for row in range(2):
        ax = axes[row][col]
        ax.hist(x[x >= 1], bins=edges, color="#4c78a8")
        ax.axvline(500, color="#d62728", lw=1.5)
        ax.set_xscale("log")
        if row == 1:
            ax.set_yscale("log")  # makes the few small films visible
            ax.set_xlabel("tokens (whitespace words), log scale")
        if col == 0:
            ax.set_ylabel("films" + (" (log scale)" if row else ""))
    axes[0][col].set_title(f"{L} ({int((x == 0).sum())} films with 0 not shown)", fontsize=10)
    axes[0][col].text(560, axes[0][col].get_ylim()[1] * 0.9, "500", color="#d62728")
fig.tight_layout()
fig.savefig(HERE / "threshold_hist.png", dpi=130)
