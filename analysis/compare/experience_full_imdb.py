"""Post-hoc robustness run (declared in docs/q2-network.md before running): experience counted from full IMDb careers.

scripts/network/film_features.py counts a person's earlier films only among the films of the movie graph, i.e.
films that later got 10,000+ IMDb votes. A person with many earlier credits there is partly a person with a
noticed past. This rebuild keeps the same teams (the graph's directors, writers, producers and first three billed
actors of each film) and counts every earlier non-adult film (titleType movie) they are credited on in IMDb's
full title.principals, in any working role: appearances as themselves ("self") and archive footage or sound don't
count, since they aren't work on the film (imdb_full_coverage.py found 343,561 "self" rows on movies). Studios and distributors aren't in IMDb, and track records (mean
ln(gross/budget) of earlier films) need money, which only graph films have: both stay as they are and are tested
unchanged, for comparison.

  Measure   <group>_experience = mean over the group's people of their earlier films (earlier release year),
            used as log(1 + x), once from the graph (film_network_features.csv) and once from full IMDb.
  Test      as the genre specialists (analysis/genre/q1_before_release.py): on the usable_money films, OLS of
            ln(gross / budget) net of the budget decile x era mean on the measure (per SD), genre dummies and the
            sequel flag, HC3 intervals; p from PERMS shuffles of the measure within first genre x budget decile,
            the model refitted every time. One measure per model, on the films where it exists.
  Check     Spearman correlation of the two versions of each count, and their medians.
What this measures: whether a team's earlier output goes with beating the budget once unnoticed films count too.
What it misses: title.principals lists each title's principal credits only (a median of 21 rows on 10,000-vote
films, 7 on unrated ones), and unrated films often have no producer or actor listed, so careers on obscure films
are undercounted; credits on films with no year can't be dated and are left out (imdb_full_coverage.py).
Outputs in analysis/compare/outputs/experience_full_imdb/. Numbers only.
Run (needs IMDb's title.principals.tsv.gz and title.basics.tsv.gz in IMDB_DIR, default DATA/imdb):
  IMDB_DATA_DIR=/path/to/data/raw IMDB_DIR=/path/to/imdb .venv/bin/python -m analysis.compare.experience_full_imdb [--perms 1000]
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.genre.q1_before_release import net_return, ols, shuffle_p
from analysis.genre.q1_genre import DATA, SEED, load, prepare

OUT = Path(__file__).resolve().parent / "outputs" / "experience_full_imdb"
PEOPLE = ("director", "writer", "producer", "cast")
COMPANIES = ("studio", "distributor")
TOP_CAST = 3
NOT_WORK = ("self", "archive_footage", "archive_sound")   # appearing as oneself or in old footage isn't work on a film


def teams(ids):
    """film -> group -> people, from the movie graph, grouped as in film_features.py."""
    e = pd.read_csv(DATA / "features" / "graph" / "edges.csv", dtype=str, usecols=["source", "target", "role", "ordering"])
    e = e[e["target"].isin(ids)]
    o = pd.to_numeric(e["ordering"], errors="coerce")
    e = e.assign(group=np.where(e["role"] == "actor", np.where(o <= TOP_CAST, "cast", ""), e["role"]))
    e = e[e["group"].isin(PEOPLE)]
    t = defaultdict(lambda: defaultdict(set))
    for s, f, g in zip(e["source"], e["target"], e["group"]):
        t[f][g].add(s)
    return t


def careers(people, imdb):
    """person -> sorted release years of every non-adult IMDb movie they hold a principal credit on."""
    tb = pd.read_csv(imdb / "title.basics.tsv.gz", sep="\t", usecols=["tconst", "titleType", "isAdult", "startYear"],
                     dtype=str, na_values="\\N", quoting=3)
    tb = tb[(tb["titleType"] == "movie") & (tb["isAdult"] != "1")]
    year = dict(zip(tb["tconst"], pd.to_numeric(tb["startYear"], errors="coerce")))
    hist = defaultdict(set)
    for ch in pd.read_csv(imdb / "title.principals.tsv.gz", sep="\t", usecols=["tconst", "nconst", "category"], dtype=str,
                          chunksize=5_000_000, quoting=3):
        ch = ch[ch["nconst"].isin(people) & ch["tconst"].isin(year.keys()) & ~ch["category"].isin(NOT_WORK)]
        for p, t in zip(ch["nconst"], ch["tconst"]):
            hist[p].add(t)
    return {p: sorted(year[t] for t in ts if year[t] == year[t]) for p, ts in hist.items()}, len(year)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    imdb = Path(os.environ.get("IMDB_DIR", DATA / "imdb"))
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    T = teams(set(f["imdb_id"]))
    people = {p for gs in T.values() for ps in gs.values() for p in ps}
    hist, n_movies = careers(people, imdb)
    nf = pd.read_csv(DATA / "features" / "film_network_features.csv", dtype={"tconst": str}).set_index("tconst")
    nf = nf.reindex(f["imdb_id"])

    cols = {}
    for g in PEOPLE:
        full = [np.mean([bisect.bisect_left(hist.get(p, []), y) for p in T[t][g]]) if T[t].get(g) else np.nan
                for t, y in zip(f["imdb_id"], f["year"])]
        cols[f"{g}_experience, full IMDb"] = np.log1p(np.array(full, float))
        cols[f"{g}_experience, 10k-vote films"] = np.log1p(nf[f"{g}_experience"].to_numpy(float))
    for g in COMPANIES:
        cols[f"{g}_experience, 10k-vote films (unchanged: not in IMDb)"] = np.log1p(nf[f"{g}_experience"].to_numpy(float))
    for g in PEOPLE + COMPANIES:
        cols[f"{g}_track, 10k-vote films (unchanged: needs money)"] = nf[f"{g}_track"].to_numpy(float)

    y_all = net_return(f)
    rows, rng = [], np.random.default_rng(SEED + 3100)
    for name, x in cols.items():
        ok = ~np.isnan(x)
        fo, Mo, y, X = f[ok].reset_index(drop=True), M[ok], y_all[ok], x[ok][:, None]
        r = ols(y, X, [name], fo, Mo, keep, label="sequel flag controlled")[0]
        grp = pd.factorize(pd.Series(Mo.argmax(1)).astype(str) + "|" + fo["budget_bin"].astype(str))[0]
        _, p = shuffle_p(y, X, 0, fo, Mo, keep, None, grp, a.perms, rng)
        r.update({"measure": name, "shuffle_p": round(p, 4)})
        rows.append(r)
        print(r, flush=True)
    res = pd.DataFrame(rows)[["measure", "n", "coef_per_sd", "lo", "hi", "p", "shuffle_p"]]

    check = []
    for g in PEOPLE:
        a10, af = cols[f"{g}_experience, 10k-vote films"], cols[f"{g}_experience, full IMDb"]
        ok = ~np.isnan(a10) & ~np.isnan(af)
        check.append({"group": g, "films": int(ok.sum()), "spearman_10k_vs_full": round(float(spearmanr(a10[ok], af[ok])[0]), 3),
                      "median_earlier_films_10k": round(float(np.median(np.expm1(a10[ok]))), 2),
                      "median_earlier_films_full": round(float(np.median(np.expm1(af[ok]))), 2)})
    chk = pd.DataFrame(check)

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "experience_tests.csv", index=False)
    chk.to_csv(OUT / "experience_check.csv", index=False)
    summ = {"films": len(f), "people_in_teams": len(people), "people_found_in_imdb": len(hist),
            "imdb_movies_non_adult": n_movies, "perms": a.perms,
            "shuffle": "measure permuted within first genre x budget decile, same OLS refitted"}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(res.to_string(index=False), chk.to_string(index=False), json.dumps(summ, indent=1), sep="\n")


if __name__ == "__main__":
    main()
