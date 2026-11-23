"""Group sizes under the shared cleaning rule, for two universes and two era binnings.

Shared rule (DECISIONS.md): analysis.clean_rule (called, not rewritten) AND no "net_gross"
flag in the budget or gross flag column AND no flag on the budget. Only tags, counts, genre, money rule outcomes and
year are used; no hit/flop OUTCOMES are looked at (group sizes follow from the quartile cut).

Universes:  (a) the 100k+ vote tier (movies_features.csv)
            (b) all of movies_features_10k.csv (the 100k+ tier plus the 10k-100k tier)
Both need a MovieSum script and are cut by the same layer rule (fewer than 500 tokens in the
dialogue or in the description layer is excluded; data_check/layer_stats.py).

Run:  python3 -I scripts-analysis/genre_counts/shared_rule_check.py
"""
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))                      # repo root, for analysis.clean_rule
import pandas as pd                                 # noqa: E402
from scipy.stats import fisher_exact                # noqa: E402
from analysis import clean_rule as cr               # noqa: E402
from genre_cells import MIN_CELL, MIN_GROUP, counts, label, pairs_for  # noqa: E402
from genre_counts import DATA, script_ids, tiny_ids  # noqa: E402
import math                                          # noqa: E402

MAIN = ["Drama", "Comedy", "Action", "Adventure", "Crime", "Thriller"]
ERAS_CURRENT = ["pre-1980", "1980s", "1990s", "2000s", "2010s", "2020s"]
ERAS_MERGED = ["1975-1989", "1990s", "2000s", "2010s", "2020s"]


def era_current(fl):
    return "pre-1980" if fl["year"] < 1980 else f"{fl['year'] // 10 * 10}s"


def era_merged(fl):
    return "1975-1989" if fl["year"] < 1990 else f"{fl['year'] // 10 * 10}s"


def load_table(name):
    with open(DATA / "features" / name, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def old_rule(r):
    """The rule used so far (genre_counts.py): both figures, positive, no infobox flag, same currency."""
    b, g = num(r["b"]), num(r["g"])
    return (b is not None and g is not None and b > 0 and g > 0 and not r["budget_flag"]
            and not r["gross_flag"] and r["budget_cur"] == r["gross_cur"])


def has_net_gross(r):
    return "net_gross" in (r["budget_flag"] or "") or "net_gross" in (r["gross_flag"] or "")


def new_rule(r):
    """analysis.clean_rule, no net_gross flag, and no flag on the budget (decided 2026-11-19)."""
    return cr.is_clean(r) and not has_net_gross(r) and not r["budget_flag"]


def analyse(name, rows, scripts, tiny, tiers=None):
    print(f"\n{'=' * 100}\nUNIVERSE {name}: {len(rows)} films in the table")
    sc = [r for r in rows if r["tconst"] in scripts]
    print(f"with a MovieSum script: {len(sc)}")
    df = pd.DataFrame(sc)
    for c in ["y", "b", "g"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    first = cr.why_not_clean(df)
    ok_new = [new_rule(r) for r in sc]
    ok_old = [old_rule(r) for r in sc]
    print(f"\nfunnel among the {len(sc)} films with a script")
    print(f"  old rule (genre_counts.py) keeps {sum(ok_old)}, drops {len(sc) - sum(ok_old)}")
    print(f"  shared rule keeps {sum(ok_new)}, drops {len(sc) - sum(ok_new)}; first reason of the films it drops:")
    for reason in list(cr.REASONS) + ["net_gross flag (extra)"]:
        if reason.startswith("net_gross"):
            n = sum(1 for r, ok, f in zip(sc, ok_new, first) if f == "clean" and has_net_gross(r))
        else:
            n = int((first == reason).sum())
        print(f"    {reason:<60}{n:>5}")
    print(f"  films kept by one rule and dropped by the other: old-only {sum(o and not n for o, n in zip(ok_old, ok_new))},"
          f" shared-only {sum(n and not o for o, n in zip(ok_old, ok_new))}")
    bf = sum(1 for r, ok in zip(sc, ok_new) if ok and r["budget_flag"] and "net_gross" not in r["budget_flag"])
    print(f"  kept by the shared rule although the BUDGET carries another flag (it only reads the gross flag): {bf}")

    keep_money = [r for r, ok in zip(sc, ok_new) if ok]
    final = [r for r in keep_money if r["tconst"] not in tiny]
    print(f"\nshared rule leaves {len(keep_money)}; tiny-layer rule drops {len(keep_money) - len(final)}; FINAL N = {len(final)}")
    old_final = [r for r, ok in zip(sc, ok_old) if ok and r["tconst"] not in tiny]
    print(f"(old rule + tiny rule gave {len(old_final)})")

    films = []
    for r in final:
        genres = [g for g in (r["genres"] or "").split(",") if g and g != "\\N"]
        y = int(float(r["y"]))
        films.append({"id": r["tconst"], "genres": genres, "year": y, "decade": y // 10 * 10,
                      "ln": math.log(float(r["g"]) / float(r["b"]))})
    films = [f for f in films if f["genres"]]
    out = []
    for binname, era_fn, eras in [("current bins", era_current, ERAS_CURRENT), ("1975-79 merged into 1975-1989", era_merged, ERAS_MERGED)]:
        pairs = pairs_for(films, "a")
        lab, exc = label(pairs, era_fn)
        c = counts(lab, exc)
        cells = Counter()
        for fl, g in pairs:
            cells[(g, era_fn(fl))] += 1
        small = {k: n for k, n in cells.items() if n < MIN_CELL}
        small_main = {k: n for k, n in small.items() if k[0] in MAIN}
        print(f"\n-- era bins: {binname}")
        print("   films per era bin (distinct films): " + ", ".join(f"{e} {sum(1 for f in films if era_fn(f) == e)}" for e in eras))
        print(f"   genre x era cells: {len(cells)} in total, {len(small)} below {MIN_CELL} films ({sum(small.values())} film-genre pairs removed)")
        print(f"   main-genre cells: {sum(1 for k in cells if k[0] in MAIN)}, below {MIN_CELL}: {len(small_main)}  -> "
              + ", ".join(f"{g} {e} ({n})" for (g, e), n in sorted(small_main.items())) if small_main else
              f"   main-genre cells: {sum(1 for k in cells if k[0] in MAIN)}, none below {MIN_CELL}")
        shown = [g for g in MAIN] + sorted([g for g in c if g not in MAIN and c[g]["hit"] >= 25], key=lambda g: -c[g]["hit"])
        print(f"   {'genre':<11}{'films':>6}{'hit':>6}{'flop':>6}{'middle':>8}{'removed':>9}   >= {MIN_GROUP} per group?")
        for g in shown:
            x = c[g]
            n = sum(1 for f in films if g in f["genres"])
            print(f"   {g:<11}{n:>6}{x['hit']:>6}{x['flop']:>6}{x['middle']:>8}{x['removed']:>9}   {'yes' if min(x['hit'], x['flop']) >= MIN_GROUP else 'NO'}")
            out.append({"universe": name, "era_bins": binname, "genre": g, "films": n, "hit": x["hit"], "flop": x["flop"],
                        "middle": x["middle"], "pairs_removed_min_cell": x["removed"]})
        print("   genres outside the main six that reach the threshold:",
              [g for g in c if g not in MAIN and min(c[g]["hit"], c[g]["flop"]) >= MIN_GROUP] or "none")
        print("   main genres below the threshold:", [g for g in MAIN if min(c[g]["hit"], c[g]["flop"]) < MIN_GROUP] or "none")
    return sc, keep_money, final, out


def main():
    scripts, tiny = script_ids(), tiny_ids()
    r100 = load_table("movies_features.csv")
    r10 = load_table("movies_features_10k.csv")
    allout = []
    _, _, f_a, o = analyse("(a) 100k+ tier only", r100, scripts, tiny)
    allout += o
    sc_b, keep_b, f_b, o = analyse("(b) all of movies_features_10k.csv", r10, scripts, tiny)
    allout += o
    with open(HERE / "shared_rule_check.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allout[0]))
        w.writeheader()
        w.writerows(allout)

    # tiny-layer rate in the less popular films (tags and counts only)
    # cross-check: this independent rebuild must give the same set as the shared table
    from final_set import final_films
    mine, shared = {r["tconst"] for r in f_b}, {f["id"] for f in final_films()}
    print(f"\ncross-check with features/film_analysis_set.csv: {len(mine)} here, {len(shared)} in the shared table, "
          f"identical: {mine == shared}")

    print(f"\n{'=' * 100}\nTINY-LAYER RATE BY TIER (films with a script; tiny = fewer than 500 tokens in dialogue or description)")
    kinds = {r["imdb_id"]: r["tiny_layer"] for r in csv.DictReader(open(HERE.parent / "data_check" / "tiny_layers.csv"))}
    ok_ids = {r["tconst"] for r in keep_b}
    print(f"{'tier':<12}{'films':>7}{'tiny':>6}{'rate':>8}   {'dialogue-poor':>14}{'description-poor':>18}   | with usable money: films, tiny, rate")
    tab = {}
    for tier in ["100k+", "10k-100k"]:
        s = [r for r in sc_b if r["tier"] == tier]
        t = [r for r in s if r["tconst"] in tiny]
        u = [r for r in s if r["tconst"] in ok_ids]
        ut = [r for r in u if r["tconst"] in tiny]
        print(f"{tier:<12}{len(s):>7}{len(t):>6}{100 * len(t) / len(s):>7.1f}%   {sum(kinds.get(r['tconst']) == 'dialogue' for r in t):>14}"
              f"{sum(kinds.get(r['tconst']) == 'description' for r in t):>18}   | {len(u)}, {len(ut)}, {100 * len(ut) / len(u):.1f}%")
        tab[tier] = (len(s), len(t), len(u), len(ut))
    a, b = tab["100k+"], tab["10k-100k"]
    print("Fisher exact, tiny rate 100k+ vs 10k-100k (all films with script): p = %.3f; (with usable money): p = %.3f"
          % (fisher_exact([[a[1], a[0] - a[1]], [b[1], b[0] - b[1]]])[1], fisher_exact([[a[3], a[2] - a[3]], [b[3], b[2] - b[3]]])[1]))
    # era of the extra films, for context
    e = Counter(("<1990" if int(float(r["y"])) < 1990 else ">=1990") for r in sc_b if r["tier"] == "10k-100k")
    e2 = Counter(("<1990" if int(float(r["y"])) < 1990 else ">=1990") for r in sc_b if r["tier"] == "100k+")
    print("share of scripted films released before 1990: 100k+ %.0f%%, 10k-100k %.0f%%" % (100 * e2["<1990"] / sum(e2.values()), 100 * e["<1990"] / sum(e.values())))


if __name__ == "__main__":
    main()
