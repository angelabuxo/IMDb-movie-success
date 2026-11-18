"""Hit / flop counts under the decided rule, for three treatments of multi-genre films.

Rule (DECISIONS.md): ln(gross/budget); quartiles cut within each genre x era cell;
hit = at or above Q3, flop = at or below Q1, middle = in between (kept in the
count, not used in the contrast). Cells with fewer than MIN_CELL films are excluded.
Era bins: before 1980, 1980s, 1990s, 2000s, 2010s, 2020s.
Only genre, money and year are used. No script text.

Treatments of a film with several genres:
  (a) it counts in every genre it has
  (b) only its FIRST-listed genre. In movies_features.csv the genres are in
      ALPHABETICAL order (checked: all 2592 multi-genre rows), so "first" means
      alphabetically first, NOT most important.
  (c) only films with exactly one genre

Run:  python3 -I scripts-analysis/genre_counts/genre_cells.py
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genre_counts import OUT, load_films  # noqa: E402

MIN_CELL = 8     # Angi: genre x era cells below this are excluded
MIN_GROUP = 50   # Angi: hit and flop groups needed for a main genre


def era(decade):
    return "pre-1980" if decade < 1980 else f"{decade}s"


ERAS = ["pre-1980", "1980s", "1990s", "2000s", "2010s", "2020s"]


def pairs_for(films, treatment):
    """List of (film, genre) pairs for treatment a, b or c."""
    out = []
    for fl in films:
        if treatment == "a":
            out += [(fl, g) for g in fl["genres"]]
        elif treatment == "b":
            out.append((fl, fl["genres"][0]))
        elif treatment == "c" and len(fl["genres"]) == 1:
            out.append((fl, fl["genres"][0]))
    return out


def label(pairs):
    """Return (labelled, excluded). labelled: list of (film, genre, era, label);
    excluded: list of (film, genre) pairs in cells below MIN_CELL."""
    cells = defaultdict(list)
    for fl, g in pairs:
        cells[(g, era(fl["decade"]))].append(fl)
    labelled, excluded = [], []
    for (g, e), grp in cells.items():
        if len(grp) < MIN_CELL:
            excluded += [(fl, g) for fl in grp]
            continue
        q1, _, q3 = statistics.quantiles([fl["ln"] for fl in grp], n=4, method="inclusive")
        for fl in grp:
            lab = "hit" if fl["ln"] >= q3 else "flop" if fl["ln"] <= q1 else "middle"
            labelled.append((fl, g, e, lab))
    return labelled, excluded


def counts(labelled, excluded):
    """Per genre: hit, middle, flop, films kept, pairs removed by MIN_CELL."""
    c = defaultdict(lambda: {"hit": 0, "middle": 0, "flop": 0, "removed": 0})
    for _, g, _, lab in labelled:
        c[g][lab] += 1
    for _, g in excluded:
        c[g]["removed"] += 1
    return c


def main():
    films = load_films()
    print(f"final analysis set: {len(films)} films")
    res = {}
    for t in "abc":
        pairs = pairs_for(films, t)
        lab, exc = label(pairs)
        res[t] = (pairs, lab, exc, counts(lab, exc))
        print(f"treatment ({t}): {len(pairs)} film-genre pairs, {len(exc)} removed by min cell size {MIN_CELL}")

    # genres with >= MIN_GROUP hits and flops under (a)
    ca = res["a"][3]
    main_genres = sorted([g for g, c in ca.items() if c["hit"] >= MIN_GROUP and c["flop"] >= MIN_GROUP],
                         key=lambda g: -ca[g]["hit"])
    print(f"\ngenres with >= {MIN_GROUP} hits and >= {MIN_GROUP} flops under (a): {main_genres}")
    print("(genres just below: " + ", ".join(
        f"{g} {c['hit']}/{c['flop']}" for g, c in sorted(ca.items(), key=lambda x: -x[1]['hit'])
        if g not in main_genres and c['hit'] >= 25) + ")")

    # 1. counts per treatment
    rows = []
    print("\n== 1. hit / middle / flop per treatment (pairs removed by min cell in brackets)")
    print(f"{'genre':<10}" + "".join(f"{'(' + t + ') hit/mid/flop [removed]':>30}" for t in "abc"))
    for g in main_genres + [g for g in ["Comedy", "Action"] if g not in main_genres]:
        line = f"{g:<10}"
        for t in "abc":
            c = res[t][3].get(g, {"hit": 0, "middle": 0, "flop": 0, "removed": 0})
            line += f"{c['hit']}/{c['middle']}/{c['flop']} [{c['removed']}]".rjust(30)
            rows.append({"treatment": t, "genre": g, **c})
        print(line)
    with open(OUT / "cells_by_treatment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["treatment", "genre", "hit", "middle", "flop", "removed"])
        w.writeheader()
        w.writerows(rows)

    # 2. overlap matrix under (a): films kept in both genres' analysis cells
    kept = defaultdict(set)
    for fl, g, _, _ in res["a"][1]:
        kept[g].add(fl["id"])
    print("\n== 2. films shared between genres (under (a), films kept after min cell size; diagonal = films in genre)")
    print(f"{'':<10}" + "".join(f"{g[:9]:>10}" for g in main_genres))
    mat = []
    for g1 in main_genres:
        print(f"{g1:<10}" + "".join(f"{len(kept[g1] & kept[g2]):>10}" for g2 in main_genres))
        mat.append({"genre": g1, **{g2: len(kept[g1] & kept[g2]) for g2 in main_genres}})
    with open(OUT / "genre_overlap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["genre"] + main_genres)
        w.writeheader()
        w.writerows(mat)

    # 3. sanity: hits and flops per era under (a), the main genres pooled
    print("\n== 3. sanity: hits and flops per era (treatment a, main genres pooled; ~equal by construction)")
    print(f"{'era':<10}{'pairs':>7}{'hit':>6}{'flop':>6}{'hit %':>8}{'flop %':>8}")
    san = []
    for e in ERAS:
        sel = [x for x in res["a"][1] if x[2] == e and x[1] in main_genres]
        n = len(sel)
        h, fl_ = sum(x[3] == "hit" for x in sel), sum(x[3] == "flop" for x in sel)
        print(f"{e:<10}{n:>7}{h:>6}{fl_:>6}{100*h/n if n else 0:>8.1f}{100*fl_/n if n else 0:>8.1f}")
        san.append({"era": e, "pairs": n, "hit": h, "flop": fl_})
    with open(OUT / "era_sanity.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["era", "pairs", "hit", "flop"])
        w.writeheader()
        w.writerows(san)
    print("\ncells below the minimum, main genres, treatment (a): (genre, era, films)")
    cells = defaultdict(int)
    for fl, g in res["a"][2]:
        cells[(g, era(fl["decade"]))] += 1
    for (g, e), n in sorted(cells.items()):
        if g in main_genres:
            print(f"  {g:<10}{e:<10}{n}")


if __name__ == "__main__":
    main()
