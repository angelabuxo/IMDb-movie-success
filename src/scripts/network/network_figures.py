"""One figure from the saved outputs of pair_returns.py, team_history.py and
movers.py (no raw data needed). Run from the repo root:

    python scripts/network/network_figures.py

Writes analysis/network/figures/network_reach.pdf and .png. Every panel uses
the no_franchise variant.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "analysis" / "network"
OUT = SRC / "figures"
VARIANT = "no_franchise"

INK, ACCENT, GREY, LIGHT, RED = "#1F2933", "#1F6F8B", "#8A969C", "#D5E3E8", "#B5482F"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                     "pdf.fonttype": 42})


def panel(ax, letter):
    ax.text(-0.3, 1.16, letter, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")


def bars_with_null(ax, labels, rows, nulls):
    x = np.arange(len(labels))
    ax.bar(x, [r["corr"] for r in rows], color=ACCENT, width=0.62, label="Observed")
    marks = {"all": ("_", INK, "Shuffle null, 95th pct."), "studio": ("x", RED, "Null within studio, 95th pct.")}
    for name in nulls:
        m, c, lab = marks[name]
        ax.scatter(x, [r[f"null_{name}_95"] for r in rows], marker=m, s=60 if m == "_" else 16, color=c, zorder=3,
                   label=lab, linewidths=1.2)
    ax.axhline(0, color=INK, lw=0.5)
    ax.set_xticks(x, labels)


def main():
    OUT.mkdir(exist_ok=True)
    pairs = json.loads((SRC / "pair_returns.json").read_text())
    g, s = pairs["graph"][VARIANT], pairs["scripts"][VARIANT]
    mv = json.loads((SRC / "movers.json").read_text())["results"][VARIANT]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0), gridspec_kw={"wspace": 0.7})
    a, b, c, d = axes

    keys = list(g["distance"])
    bars_with_null(a, keys, [g["distance"][k] for k in keys], ["all"])
    a.set_xlabel("Steps apart in the film network")
    a.set_ylabel("Return correlation of film pairs")
    a.set_title("Only direct links")
    panel(a, "A")

    keys = list(g["dose"])
    bars_with_null(b, keys, [g["dose"][k] for k in keys], ["all", "studio"])
    b.set_xlabel("People two films share")
    b.set_title("More shared people, more alike")
    b.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.45, -0.3), handlelength=1.2)
    panel(b, "B")

    dec = s["unlinked_by_cosine_decile"]
    x = np.arange(1, 11)
    c.fill_between(x, [r["null_all_mean"] * 2 - r["null_all_95"] for r in dec], [r["null_all_95"] for r in dec],
                   color=LIGHT, lw=0, label="Shuffle null")
    c.plot(x, [r["corr"] for r in dec], "o-", color=GREY, ms=2.5, lw=0.8, label="No shared person")
    link = s["pairs"]["linked"]
    c.axhline(link["corr"], color=ACCENT, lw=1.2, label="Share a person")
    c.axhline(0, color=INK, lw=0.5)
    c.set_xticks([1, 5, 10])
    c.set_xlabel("Script similarity decile")
    c.set_title("Similar scripts, no link")
    c.legend(frameon=False, fontsize=5.5, loc="center left", bbox_to_anchor=(0.0, 0.62), handlelength=1.2)
    panel(c, "C")

    roles = ["director", "writer", "producer", "cast"]
    y = np.arange(len(roles))[::-1]
    for off, group, face in ((0.17, "stayers", "white"), (-0.17, "movers", None)):
        for kind, col in (("person", ACCENT), ("studio", RED)):
            est = [mv[r][group][kind] for r in roles]
            lo = [mv[r][group][kind] - mv[r][group][f"{kind}_ci"][0] for r in roles]
            hi = [mv[r][group][f"{kind}_ci"][1] - mv[r][group][kind] for r in roles]
            d.errorbar(est, y + off + (0.06 if kind == "person" else -0.06), xerr=[lo, hi], fmt="o", ms=3, color=col,
                       mfc=face or col, lw=0.7, label=f"{kind.capitalize()}'s record, {group}")
    d.axvline(0, color=INK, lw=0.5)
    d.set_yticks(y, [r.capitalize() for r in roles])
    d.set_xlabel("Effect on the film's return (SD per SD)")
    d.set_title("Records stay with the studio")
    d.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.3, -0.3), ncol=2, handletextpad=0.2,
             columnspacing=0.8)
    panel(d, "D")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"network_reach.{ext}", bbox_inches="tight", dpi=300)
    print(f"wrote {OUT / 'network_reach.pdf'} and .png")


if __name__ == "__main__":
    main()
