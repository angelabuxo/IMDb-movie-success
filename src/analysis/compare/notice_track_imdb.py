"""Post-hoc robustness run (declared in src/analysis/network/Q2_NETWORK.md before running): a track record from IMDb alone.

Money track records (mean ln(gross/budget) of earlier films) need budget and gross, which we have only for films with
10,000+ IMDb votes, so a person's unnoticed flops never enter them. IMDb has a trace of every film: its votes and its
rating. This builds two IMDb-only track records over each person's full career and tests them like the money ones.

  Teams     as experience_full_imdb.py: the graph's directors, writers, producers and first three billed actors of
            each usable_money film. Careers: every non-adult IMDb movie they hold a working principal credit on
            (not "self" or archive footage), released in an earlier year than the film. Films with no year are left out.
  Measures  <group>_unnoticed = mean over the group's people (with at least one earlier film) of the share of their
            earlier films with fewer than UNNOTICED votes (no rating counts as fewer);
            <group>_rating = mean over the group's people of the mean IMDb rating of their earlier rated films.
  Test      as experience_full_imdb.py: OLS of ln(gross / budget) net of the budget decile x era mean on the measure
            (per SD), genre dummies and the sequel flag; p from PERMS shuffles within first genre x budget decile.
  Check     Spearman correlation of each measure with the group's money track record (film_network_features.csv),
            on films that have both: a second method for the same idea.
What this measures: whether a team whose earlier films went unnoticed, or were rated low, does worse with its budget.
What it misses: votes and ratings are counted today, not when the film was made, so a later revival counts as notice;
votes and ratings are not money; unrated films often list no producer or actor in title.principals
(imdb_full_coverage.py), so obscure careers are undercounted.
Outputs in analysis/compare/outputs/notice_track_imdb/. Numbers only.
Run (needs title.principals, title.basics and title.ratings in IMDB_DIR):
  IMDB_DATA_DIR=/path/to/data/raw IMDB_DIR=/path/to/imdb .venv/bin/python -m analysis.compare.notice_track_imdb [--perms 1000]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.compare.experience_full_imdb import NOT_WORK, PEOPLE, teams
from analysis.genre.q1_before_release import net_return, ols, shuffle_p
from analysis.genre.q1_genre import DATA, SEED, load, prepare

OUT = Path(__file__).resolve().parent / "outputs" / "notice_track_imdb"
UNNOTICED = 1000


def careers(people, imdb):
    """person -> list of (year, votes, rating) for every dated non-adult IMDb movie they hold a working credit on."""
    tb = pd.read_csv(imdb / "title.basics.tsv.gz", sep="\t", usecols=["tconst", "titleType", "isAdult", "startYear"],
                     dtype=str, na_values="\\N", quoting=3)
    tb = tb[(tb["titleType"] == "movie") & (tb["isAdult"] != "1")]
    tb["year"] = pd.to_numeric(tb["startYear"], errors="coerce")
    tb = tb.dropna(subset=["year"])
    rt = pd.read_csv(imdb / "title.ratings.tsv.gz", sep="\t", dtype={"tconst": str}, quoting=3).set_index("tconst")
    info = pd.DataFrame({"year": tb["year"].to_numpy()}, index=tb["tconst"])
    info["votes"] = rt["numVotes"].reindex(info.index).fillna(0).to_numpy()
    info["rating"] = rt["averageRating"].reindex(info.index).to_numpy()
    info = dict(zip(info.index, info.itertuples(index=False, name=None)))
    hist = defaultdict(set)
    for ch in pd.read_csv(imdb / "title.principals.tsv.gz", sep="\t", usecols=["tconst", "nconst", "category"], dtype=str,
                          chunksize=5_000_000, quoting=3):
        ch = ch[ch["nconst"].isin(people) & ch["tconst"].isin(info.keys()) & ~ch["category"].isin(NOT_WORK)]
        for p, t in zip(ch["nconst"], ch["tconst"]):
            hist[p].add(t)
    return {p: [info[t] for t in ts] for p, ts in hist.items()}, len(info)


def person(h, y):
    """share of earlier films under UNNOTICED votes, and their mean rating, for one person before year y."""
    e = [(v, r) for yr, v, r in h if yr < y]
    if not e:
        return np.nan, np.nan
    r = [x for _, x in e if x == x]
    return np.mean([v < UNNOTICED for v, _ in e]), (np.mean(r) if r else np.nan)


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
    nf = pd.read_csv(DATA / "features" / "film_network_features.csv", dtype={"tconst": str}).set_index("tconst").reindex(f["imdb_id"])

    cols, earlier = {}, []
    for g in PEOPLE:
        un, ra = [], []
        for t, y in zip(f["imdb_id"], f["year"]):
            vals = [person(hist.get(p, []), y) for p in T[t].get(g, ())]
            u = [v[0] for v in vals if v[0] == v[0]]
            r = [v[1] for v in vals if v[1] == v[1]]
            un.append(np.mean(u) if u else np.nan)
            ra.append(np.mean(r) if r else np.nan)
        cols[f"{g}_unnoticed"] = np.array(un, float)
        cols[f"{g}_rating"] = np.array(ra, float)
    for g in PEOPLE:
        cols[f"{g}_track (money, 10k-vote films)"] = nf[f"{g}_track"].to_numpy(float)

    y_all = net_return(f)
    rows, rng = [], np.random.default_rng(SEED + 3200)
    for name, x in cols.items():
        ok = ~np.isnan(x)
        fo, Mo, y, X = f[ok].reset_index(drop=True), M[ok], y_all[ok], x[ok][:, None]
        r = ols(y, X, [name], fo, Mo, keep, label="sequel flag controlled")[0]
        grp = pd.factorize(pd.Series(Mo.argmax(1)).astype(str) + "|" + fo["budget_bin"].astype(str))[0]
        _, p = shuffle_p(y, X, 0, fo, Mo, keep, None, grp, a.perms, rng)
        r.update({"measure": name, "shuffle_p": round(p, 4), "median": round(float(np.median(x[ok])), 3)})
        rows.append(r)
        print(r, flush=True)
    res = pd.DataFrame(rows)[["measure", "n", "median", "coef_per_sd", "lo", "hi", "p", "shuffle_p"]]

    check = []
    for g in PEOPLE:
        m = cols[f"{g}_track (money, 10k-vote films)"]
        for k in ("unnoticed", "rating"):
            x = cols[f"{g}_{k}"]
            ok = ~np.isnan(x) & ~np.isnan(m)
            check.append({"group": g, "measure": k, "films_with_both": int(ok.sum()),
                          "spearman_with_money_track": round(float(spearmanr(x[ok], m[ok])[0]), 3)})
    chk = pd.DataFrame(check)

    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "notice_tests.csv", index=False)
    chk.to_csv(OUT / "notice_check.csv", index=False)
    summ = {"films": len(f), "people_in_teams": len(people), "people_found_in_imdb": len(hist),
            "imdb_movies_dated_non_adult": n_movies, "unnoticed_votes_below": UNNOTICED, "perms": a.perms,
            "shuffle": "measure permuted within first genre x budget decile, same OLS refitted"}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    with pd.option_context("display.width", 200, "display.max_colwidth", 70):
        print(res.to_string(index=False), chk.to_string(index=False), json.dumps(summ, indent=1), sep="\n")


if __name__ == "__main__":
    main()
