"""Is IMDb's full movie list a sound base for counting careers and building the people network?

experience_full_imdb.py and network_communities_full.py extend careers and the network from the 12,673 films with
10,000+ votes to every non-adult film IMDb lists (titleType movie). This describes that wider list and checks the
one file both lean on, title.principals, which holds only each title's principal credits.

  Size       non-adult movies, how many have a year, how many are dated in the future, per decade.
  Noticed    how many have any rating at all, and at 10 / 100 / 1,000 / 10,000+ votes (title.ratings).
  Credits    per movie: is there a director, a producer, an actor in title.principals; how many principal rows a
             title has and how many sit at the cap (the largest ordering seen), which would mean a cut list.
  Second     title.crew lists every director and writer of a title. Share of those found among the director and
  source     writer rows of title.principals, by vote tier: a direct check of principals' completeness for two roles.
  Careers    for the people on the usable_money films' teams (as experience_full_imdb.py builds them): what their
             full-IMDb credits are on, by the earlier film's vote tier, so it's clear what the wider count adds.
  Hand check 25 random team people with their counts from both sources, to check against IMDb's website by hand
             (hand_check_sheet.csv; the group fills in the last column).
What this captures: coverage and internal consistency of the IMDb files. What it misses: whether IMDb itself has
every film a person made (no second full source here), and credits beyond the principal ones.
Outputs in analysis/compare/outputs/imdb_full_coverage/ (numbers, and names for the hand-check sheet only).
Run (HPC: about 4 minutes): IMDB_DATA_DIR=/path/to/data/raw IMDB_DIR=/path/to/imdb .venv/bin/python -m analysis.compare.imdb_full_coverage
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.compare.experience_full_imdb import PEOPLE, teams
from analysis.genre.q1_genre import DATA, SEED, load, prepare

OUT = Path(__file__).resolve().parent / "outputs" / "imdb_full_coverage"
TIERS = [(None, None, "no rating"), (1, 10, "1-9 votes"), (10, 100, "10-99"), (100, 1_000, "100-999"),
         (1_000, 10_000, "1,000-9,999"), (10_000, np.inf, "10,000+")]
SAMPLE = 25


def tier(v):
    v = np.nan_to_num(np.asarray(v, float), nan=0.0)
    out = np.full(len(v), "no rating", dtype=object)
    for lo, hi, name in TIERS[1:]:
        out[(v >= lo) & (v < hi)] = name
    return out


def main():
    imdb = Path(os.environ.get("IMDB_DIR", DATA / "imdb"))
    tb = pd.read_csv(imdb / "title.basics.tsv.gz", sep="\t", usecols=["tconst", "titleType", "isAdult", "startYear", "genres"],
                     dtype=str, na_values="\\N", quoting=3)
    mv = tb[(tb["titleType"] == "movie") & (tb["isAdult"] != "1")].copy()
    mv["year"] = pd.to_numeric(mv["startYear"], errors="coerce")
    rt = pd.read_csv(imdb / "title.ratings.tsv.gz", sep="\t", dtype={"tconst": str}, quoting=3)
    mv["votes"] = mv["tconst"].map(rt.set_index("tconst")["numVotes"])
    mv["tier"] = tier(mv["votes"].to_numpy())
    ids = set(mv["tconst"])
    out = {}

    out["size"] = {"non_adult_movies": len(mv), "with_year": int(mv["year"].notna().sum()),
                   "dated_after_2026": int((mv["year"] > 2026).sum()),
                   "genre_missing": int(mv["genres"].isna().sum()),
                   "documentary": int(mv["genres"].fillna("").str.contains("Documentary").sum())}
    dec = (mv["year"] // 10 * 10).value_counts().sort_index()
    tiers = mv["tier"].value_counts().reindex([t[2] for t in TIERS]).fillna(0).astype(int)

    # principals: who is credited on each movie, and is the list cut
    ch = []
    for c in pd.read_csv(imdb / "title.principals.tsv.gz", sep="\t", usecols=["tconst", "ordering", "nconst", "category"],
                         dtype=str, chunksize=5_000_000, quoting=3):
        ch.append(c[c["tconst"].isin(ids)])
    pr = pd.concat(ch, ignore_index=True)
    pr["ordering"] = pd.to_numeric(pr["ordering"], errors="coerce")
    cats = pr["category"].value_counts()
    rows = pr.groupby("tconst").agg(rows=("ordering", "size"), max_ord=("ordering", "max"))
    cap = int(rows["max_ord"].max())
    has = {r: set(pr.loc[pr["category"].isin(c), "tconst"]) for r, c in
           (("director", ["director"]), ("producer", ["producer"]), ("actor", ["actor", "actress"]), ("writer", ["writer"]))}
    credit = []
    for t in [x[2] for x in TIERS]:
        m = set(mv.loc[mv["tier"] == t, "tconst"])
        r = rows.reindex(list(m))
        credit.append({"vote_tier": t, "movies": len(m), "with_any_principal": round(r["rows"].notna().mean(), 3),
                       **{f"with_{k}": round(len(m & v) / max(len(m), 1), 3) for k, v in has.items()},
                       "median_principal_rows": float(r["rows"].median()) if r["rows"].notna().any() else None,
                       f"share_at_cap_{cap}": round(float((r["max_ord"] >= cap).mean()), 3)})
    out["principals"] = {"rows_on_movies": len(pr), "cap_ordering": cap, "categories": {k: int(v) for k, v in cats.items()}}

    # second source: title.crew directors and writers, found among principals' rows of the same role
    cr = pd.read_csv(imdb / "title.crew.tsv.gz", sep="\t", dtype=str, na_values="\\N", quoting=3)
    cr = cr[cr["tconst"].isin(ids)]
    agree = []
    pairs_p = {r: set(zip(pr.loc[pr["category"] == r, "tconst"], pr.loc[pr["category"] == r, "nconst"])) for r in ("director", "writer")}
    vt = mv.set_index("tconst")["tier"]
    for role, col in (("director", "directors"), ("writer", "writers")):
        e = cr[["tconst", col]].dropna().assign(nconst=lambda d: d[col].str.split(",")).explode("nconst")
        e["found"] = [(t, n) in pairs_p[role] for t, n in zip(e["tconst"], e["nconst"])]
        e["tier"] = e["tconst"].map(vt)
        for t, g in e.groupby("tier"):
            agree.append({"role": role, "vote_tier": t, "crew_credits": len(g), "share_in_principals": round(g["found"].mean(), 3)})
        agree.append({"role": role, "vote_tier": "all", "crew_credits": len(e), "share_in_principals": round(e["found"].mean(), 3)})

    # careers of the usable films' teams: what the full count adds
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    T = teams(set(f["imdb_id"]))
    team = {(p, g) for gs in T.values() for g, ps in gs.items() for p in ps}
    people = {p for p, _ in team}
    hp = pr[pr["nconst"].isin(people)].drop_duplicates(["nconst", "tconst"])
    hp = hp.assign(tier=hp["tconst"].map(vt), year=hp["tconst"].map(mv.set_index("tconst")["year"]))
    car = hp["tier"].value_counts(normalize=True).reindex([t[2] for t in TIERS]).fillna(0).round(3)
    out["careers"] = {"team_people": len(people), "people_with_credits": int(hp["nconst"].nunique()),
                      "credits": len(hp), "credits_without_year": int(hp["year"].isna().sum()),
                      "share_of_credits_by_vote_tier": car.to_dict(),
                      "share_on_10k_vote_films": float(car.get("10,000+", 0))}

    # hand-check sheet: random team people, their total credited movies in full IMDb and in the 10k graph
    e = pd.read_csv(DATA / "features" / "graph" / "edges.csv", dtype=str, usecols=["source", "target"])
    g10 = e.groupby("source")["target"].nunique()
    name = pd.read_csv(DATA / "features" / "graph" / "nodes.csv", usecols=["id", "label"]).set_index("id")["label"]
    rng = np.random.default_rng(SEED + 4242)
    pick = sorted(team)
    pick = [pick[i] for i in rng.choice(len(pick), SAMPLE, replace=False)]
    full_n = hp.groupby("nconst")["tconst"].nunique()
    sheet = pd.DataFrame([{"nconst": p, "name": name.get(p, ""), "team_role": g, "movies_full_imdb_principals": int(full_n.get(p, 0)),
                           "movies_in_10k_graph": int(g10.get(p, 0)),
                           "movies_on_imdb_website (fill in by hand: feature films in their filmography)": ""} for p, g in pick])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame({"decade": dec.index.astype(int), "movies": dec.values}).to_csv(OUT / "movies_by_decade.csv", index=False)
    pd.DataFrame({"vote_tier": tiers.index, "movies": tiers.values}).to_csv(OUT / "movies_by_vote_tier.csv", index=False)
    pd.DataFrame(credit).to_csv(OUT / "principal_credits_by_tier.csv", index=False)
    pd.DataFrame(agree).to_csv(OUT / "crew_vs_principals.csv", index=False)
    sheet.to_csv(OUT / "hand_check_sheet.csv", index=False)
    with pd.option_context("display.width", 220):
        print(json.dumps(out, indent=1), tiers.to_string(), dec.to_string(), pd.DataFrame(credit).to_string(index=False),
              pd.DataFrame(agree).to_string(index=False), sheet.iloc[:, :5].to_string(index=False), sep="\n\n")


if __name__ == "__main__":
    main()
