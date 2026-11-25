"""Post-hoc check (declared in src/analysis/network/Q2_NETWORK.md before running): how far could missing flops inflate track records?

A track record (scripts/network/film_features.py) is the mean ln(gross / budget) of a group's earlier clean
films. Money exists only for the movie graph's films (10,000+ IMDb votes), so a person's unnoticed earlier films,
mostly flops, never enter it. This script rebuilds the people groups' track records (directors, writers,
producers, first three billed actors) and asks two things, with the test of experience_full_imdb.py (OLS per SD
on the return net of budget decile x era, genre dummies and sequel flag; 1,000 shuffles of the measure within
first genre x budget decile, the model refitted each time):

  (a) Bound   every earlier IMDb film of the group (title.principals, non-adult movies, any category) that has no
              clean money figure is given a bad return v and pooled with the clean ones:
                track_v = (sum of clean earlier returns + v x number of earlier films without money) / all earlier
              v = the mean return of films Wikipedia lacks that TMDB has money for (analysis/genre/outputs/tmdb), and
              v = the mean of the bottom quarter of all clean returns. Same films as the original track record (at
              least two clean earlier films). If track_v still goes with beating the budget, missing flops can't be
              what drives the track record.
  (b) Split   the original track record tested separately on films whose group's earlier films mostly have money
              (coverage = clean / all earlier films, at or above the group's median) and on the rest. If missing
              flops inflate it, the effect should be larger where coverage is low.
What this measures: whether track records survive a pessimistic guess about what is missing. What it misses: the
true returns of unnoticed films; a constant v also mixes how many films a person made into the record.
Outputs in analysis/compare/outputs/track_record_bounds/. Numbers only.
Run: IMDB_DATA_DIR=/path/to/data/raw IMDB_DIR=/path/to/imdb .venv/bin/python -m analysis.compare.track_record_bounds [--perms 1000]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.genre.q1_before_release import net_return, ols, shuffle_p
from analysis.genre.q1_genre import DATA, ROOT, SEED, load, prepare

OUT = Path(__file__).resolve().parent / "outputs" / "track_record_bounds"
PEOPLE = ("director", "writer", "producer", "cast")
TOP_CAST = 3


def graph():
    """Clean ln multiple and year of every graph movie, and each person's groups on each graph movie."""
    n = pd.read_csv(DATA / "features" / "graph" / "nodes.csv", dtype={"id": str}, low_memory=False,
                    usecols=["id", "type", "year", "multiple", "clean"])
    n = n[(n["type"] == "movie") & n["year"].notna()]
    year = dict(zip(n["id"], n["year"].astype(int)))
    cl = n[(n["clean"] == 1) & n["multiple"].notna()]
    lm = dict(zip(cl["id"], np.log(cl["multiple"].astype(float))))
    e = pd.read_csv(DATA / "features" / "graph" / "edges.csv", dtype=str, usecols=["source", "target", "role", "ordering"])
    e = e[e["target"].isin(year.keys())]
    o = pd.to_numeric(e["ordering"], errors="coerce")
    e = e.assign(group=np.where(e["role"] == "actor", np.where(o <= TOP_CAST, "cast", ""), e["role"]))
    e = e[e["group"].isin(PEOPLE)]
    return year, lm, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    imdb = Path(os.environ.get("IMDB_DIR", DATA / "imdb"))
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    year, lm, e = graph()
    ids = set(f["imdb_id"])
    team = defaultdict(lambda: defaultdict(set))
    for s, t, g in zip(e["source"], e["target"], e["group"]):
        if t in ids:
            team[t][g].add(s)
    people = {p for gs in team.values() for ps in gs.values() for p in ps}
    clean_hist = defaultdict(set)                # person -> clean graph films they hold a group credit on
    for s, t in zip(e["source"], e["target"]):
        if s in people and t in lm:
            clean_hist[s].add(t)

    tb = pd.read_csv(imdb / "title.basics.tsv.gz", sep="\t", usecols=["tconst", "titleType", "isAdult", "startYear"],
                     dtype=str, na_values="\\N", quoting=3)
    tb = tb[(tb["titleType"] == "movie") & (tb["isAdult"] != "1")]
    iyear = dict(zip(tb["tconst"], pd.to_numeric(tb["startYear"], errors="coerce")))
    full = defaultdict(set)                      # person -> every IMDb movie they hold a principal credit on
    for ch in pd.read_csv(imdb / "title.principals.tsv.gz", sep="\t", usecols=["tconst", "nconst"], dtype=str,
                          chunksize=5_000_000, quoting=3):
        ch = ch[ch["nconst"].isin(people) & ch["tconst"].isin(iyear.keys())]
        for p, t in zip(ch["nconst"], ch["tconst"]):
            full[p].add(t)

    clean_all = np.array(list(lm.values()))
    v_low = float(clean_all[clean_all <= np.quantile(clean_all, .25)].mean())
    v_tmdb = json.loads((ROOT / "src/analysis/genre/outputs/tmdb/summary.json").read_text())["money"]["mean_ln_ratio_wikipedia_lacks"]
    nf = pd.read_csv(DATA / "features" / "film_network_features.csv", dtype={"tconst": str}).set_index("tconst").reindex(f["imdb_id"])

    cols, cover = {}, {}
    for g in PEOPLE:
        orig, b_t, b_l, cov = [], [], [], []
        for t, y in zip(f["imdb_id"], f["year"]):
            P = team[t].get(g, set())
            vals = [lm[x] for p in P for x in clean_hist[p] if year[x] < y]          # pooled, as film_features.py
            nm = sum(1 for p in P for x in full[p] if x not in lm and iyear[x] == iyear[x] and iyear[x] < y)
            if len(vals) < 2:
                orig.append(np.nan); b_t.append(np.nan); b_l.append(np.nan); cov.append(np.nan)
                continue
            s, k = sum(vals), len(vals)
            orig.append(s / k)
            b_t.append((s + v_tmdb * nm) / (k + nm))
            b_l.append((s + v_low * nm) / (k + nm))
            cov.append(k / (k + nm))
        cols[f"{g}_track, rebuilt"] = np.array(orig)
        cols[f"{g}_track, films without money at {v_tmdb:+.3f} (TMDB gap mean)"] = np.array(b_t)
        cols[f"{g}_track, films without money at {v_low:+.3f} (bottom-quarter mean)"] = np.array(b_l)
        cover[g] = np.array(cov)

    y_all = net_return(f)
    rng = np.random.default_rng(SEED + 3300)

    def test(name, x, label="all"):
        ok = ~np.isnan(x)
        fo, Mo, y, X = f[ok].reset_index(drop=True), M[ok], y_all[ok], x[ok][:, None]
        r = ols(y, X, [name], fo, Mo, keep, label=label)[0]
        grp = pd.factorize(pd.Series(Mo.argmax(1)).astype(str) + "|" + fo["budget_bin"].astype(str))[0]
        _, p = shuffle_p(y, X, 0, fo, Mo, keep, None, grp, a.perms, rng)
        r.update({"measure": name, "films_subset": label, "shuffle_p": round(p, 4)})
        print(r, flush=True)
        return r

    rows, check = [], []
    for name, x in cols.items():
        rows.append(test(name, x))
    for g in PEOPLE:
        x, c = cols[f"{g}_track, rebuilt"], cover[g]
        med = float(np.nanmedian(c))
        hi, lo = np.where(c >= med, x, np.nan), np.where(c < med, x, np.nan)
        rows.append(test(f"{g}_track, rebuilt", hi, f"coverage >= median ({med:.2f})"))
        rows.append(test(f"{g}_track, rebuilt", lo, f"coverage < median ({med:.2f})"))
        ok = ~np.isnan(x) & nf[f"{g}_track"].notna().to_numpy()
        check.append({"group": g, "films_with_track": int((~np.isnan(x)).sum()),
                      "corr_with_film_features_track": round(float(np.corrcoef(x[ok], nf[f"{g}_track"].to_numpy()[ok])[0, 1]), 4),
                      "median_coverage": round(med, 3), "q25_coverage": round(float(np.nanquantile(c, .25)), 3),
                      "q75_coverage": round(float(np.nanquantile(c, .75)), 3)})
    res = pd.DataFrame(rows)[["measure", "films_subset", "n", "coef_per_sd", "lo", "hi", "p", "shuffle_p"]]
    chk = pd.DataFrame(check)
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "track_bounds.csv", index=False)
    chk.to_csv(OUT / "track_check.csv", index=False)
    summ = {"films": len(f), "v_tmdb_gap_mean": v_tmdb, "v_bottom_quarter_mean": round(v_low, 4),
            "clean_graph_films": len(lm), "perms": a.perms}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    with pd.option_context("display.width", 220, "display.max_colwidth", 80):
        print(res.to_string(index=False), chk.to_string(index=False), json.dumps(summ, indent=1), sep="\n")


if __name__ == "__main__":
    main()
