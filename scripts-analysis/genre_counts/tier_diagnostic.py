"""Tier diagnostic (counts only, no text): how many hits and flops are 100k+ vote films?

On the final set (all films of the 10k+ table that pass the shared rules), with the decided labels:
ln(gross/budget), quartiles cut within genre x era cell (current era bins), cells under 8 films excluded,
hit = top quarter, flop = bottom quarter, middle half kept only as a baseline. A film counts once in each
of its genres (treatment (a)), so the pooled lines count film-genre pairs.

It shows whether the vote tier is unevenly spread between hits and flops. It does not say why, and it
decides nothing: whether it needs a control was decided later by the group (D10).

Run:  python3 -I scripts-analysis/genre_counts/tier_diagnostic.py     (writes tier_diagnostic.csv)
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from final_set import final_films  # noqa: E402
from genre_cells import label, pairs_for  # noqa: E402

MAIN = ["Drama", "Comedy", "Action", "Adventure", "Crime", "Thriller"]


def main():
    films = final_films()
    tier = {f["id"]: f["tier"] for f in films}
    n100 = sum(1 for f in films if f["tier"] == "100k+")
    print(f"final set: {len(films)} films, {n100} at 100k+ ({100 * n100 / len(films):.1f} %), {len(films) - n100} at 10k-100k")
    lab, exc = label(pairs_for(films, "a"))
    print(f"film-genre pairs in cells of at least 8 films: {len(lab)} ({len(exc)} removed by the cell minimum)\n")

    groups = {"pooled, six main genres": lambda g: g in MAIN, "pooled, all genres": lambda g: True}
    rows, lines = [], []
    def add(name, sel):
        c = defaultdict(lambda: [0, 0])    # label -> [100k+ count, total]
        for fl, g, _, lb in lab:
            if sel(g):
                c[lb][1] += 1
                c[lb][0] += tier[fl["id"]] == "100k+"
                c["all"][1] += 1
                c["all"][0] += tier[fl["id"]] == "100k+"
        row = {"group": name}
        for lb in ["hit", "flop", "middle", "all"]:
            row[f"{lb}_100k"], row[f"{lb}_total"] = c[lb]
            row[f"{lb}_share_100k_pct"] = round(100 * c[lb][0] / c[lb][1], 1) if c[lb][1] else ""
        rows.append(row)

    for g in MAIN:
        add(g, lambda x, g=g: x == g)
    for name, sel in groups.items():
        add(name, sel)
    print(f"{'':<25}{'hits 100k+/all':>18}{'flops 100k+/all':>18}{'middle 100k+/all':>19}{'all cells':>16}")
    for r in rows:
        cell = lambda lb: f"{r[lb + '_100k']}/{r[lb + '_total']} ({r[lb + '_share_100k_pct']}%)"
        print(f"{r['group']:<25}{cell('hit'):>18}{cell('flop'):>18}{cell('middle'):>19}{cell('all'):>16}")
    with open(HERE / "tier_diagnostic.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
