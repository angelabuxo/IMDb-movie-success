"""Network features for each movie, built only from films released in earlier
years, so a film's own result can't leak into them.

Run from the data folder after build_graph.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/film_features.py

Groups: studio, distributor, producer, writer, director, cast (top three
billed). For each movie and group:
- <group>_track: mean ln(gross / budget) of the group's earlier clean films,
  with <group>_track_n the number of those films (blank under two);
- <group>_experience: mean number of earlier films in the graph, any role.
Also:
- cast_reteamed: share of the top-three cast who acted in an earlier film by
  one of this film's directors;
- studio_prior_budget: mean ln(budget) of the studios' earlier clean films,
  and studio_prior_horror: the horror share of their earlier films, to tell a
  studio's record apart from a cheap-genre habit;
- people_pagerank_mean / _max and people_degree_mean: PageRank and degree of
  the film's key people (directors, writers, producers, top-five cast) in a
  collaboration network of films from earlier years, where two people are
  linked by the number of earlier films they share;
- company_pagerank_mean: the same for its studios and distributors in a
  company network of earlier films.

The collaboration networks are rebuilt for every release year from 1950,
with igraph's weighted PageRank. Writes features/film_network_features.csv.
"""

import bisect
import csv
import math
from collections import defaultdict
from itertools import combinations

import igraph as ig
import numpy as np

NODES = "features/graph/nodes.csv"
EDGES = "features/graph/edges.csv"
OUT = "features/film_network_features.csv"
FIRST_YEAR = 1950
GROUPS = ["studio", "distributor", "producer", "writer", "director", "cast"]


def group(role, ordering, top):
    if role == "actor":
        return "cast" if ordering and ordering <= top else None
    return role if role in GROUPS else None


def main():
    movie = {}
    with open(NODES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["type"] == "movie" and r["year"]:
                m = float(r["multiple"]) if r["multiple"] else None
                movie[r["id"]] = {"year": int(r["year"]), "clean": r["clean"] == "1",
                                  "log_mult": math.log(m) if m and r["clean"] == "1" else None,
                                  "log_budget": math.log(float(r["budget_usd"])) if r["budget_usd"] and r["clean"] == "1" else None,
                                  "horror": "Horror" in (r["genres"] or "")}
    credits = defaultdict(lambda: defaultdict(set))   # film -> group -> entities
    key_people, companies = defaultdict(set), defaultdict(set)
    hist = defaultdict(list)                          # entity -> sorted years of all films
    with open(EDGES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            t = r["target"]
            if t not in movie:
                continue
            o = int(r["ordering"]) if r["ordering"] else None
            g = group(r["role"], o, 3)
            if g:
                credits[t][g].add(r["source"])
            if r["role"] in ("director", "writer", "producer") or (r["role"] == "actor" and o and o <= 5):
                key_people[t].add(r["source"])
            if r["role"] in ("studio", "distributor"):
                companies[t].add(r["source"])
            hist[r["source"]].append(movie[t]["year"])
    for v in hist.values():
        v.sort()
    clean_hist = defaultdict(list)                    # entity -> [(year, log_mult, log_budget, horror)]
    for t, gs in credits.items():
        if movie[t]["log_mult"] is not None:
            for ents in gs.values():
                for e in ents:
                    clean_hist[e].append((movie[t]["year"], movie[t]["log_mult"], movie[t]["log_budget"], movie[t]["horror"]))
    for v in clean_hist.values():
        v.sort()
    clean_years = {e: [x[0] for x in v] for e, v in clean_hist.items()}
    dir_films, act_films = defaultdict(set), defaultdict(set)
    for t, gs in credits.items():
        for d in gs.get("director", ()):
            dir_films[d].add(t)
    with open(EDGES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["role"] == "actor" and r["target"] in movie:
                act_films[r["source"]].add(r["target"])

    def earlier(e, y):
        return clean_hist.get(e, [])[:bisect.bisect_left(clean_years.get(e, []), y)]

    rows = {}
    for t in sorted(movie):
        y = movie[t]["year"]
        row = {"tconst": t, "year": y}
        for g in GROUPS:
            ents = credits[t].get(g, set())
            prev = [x for e in ents for x in earlier(e, y)]
            row[f"{g}_track_n"] = len(prev)
            row[f"{g}_track"] = float(np.mean([x[1] for x in prev])) if len(prev) >= 2 else None
            row[f"{g}_experience"] = float(np.mean([bisect.bisect_left(hist[e], y) for e in ents])) if ents else None
            if g == "studio":
                row["studio_prior_budget"] = float(np.mean([x[2] for x in prev])) if len(prev) >= 2 else None
                row["studio_prior_horror"] = float(np.mean([x[3] for x in prev])) if len(prev) >= 2 else None
        ds, cs = credits[t].get("director", set()), credits[t].get("cast", set())
        if ds and cs:
            before = {f for d in ds for f in dir_films[d] if movie[f]["year"] < y}
            row["cast_reteamed"] = float(np.mean([bool(act_films[c] & before) for c in cs]))
        else:
            row["cast_reteamed"] = None
        rows[t] = row

    # Rolling collaboration networks: for year Y, films from years before Y only.
    by_year = defaultdict(list)
    for t in movie:
        by_year[movie[t]["year"]].append(t)
    pw, cw = defaultdict(int), defaultdict(int)       # pair -> shared earlier films
    for t in movie:
        if movie[t]["year"] < FIRST_YEAR:
            for a, b in combinations(sorted(key_people[t]), 2):
                pw[(a, b)] += 1
            for a, b in combinations(sorted(companies[t]), 2):
                cw[(a, b)] += 1
    for y in sorted(k for k in by_year if k >= FIRST_YEAR):
        for weights, members, prefix in ((pw, key_people, "people"), (cw, companies, "company")):
            names = sorted({n for pair in weights for n in pair})
            idx = {n: i for i, n in enumerate(names)}
            pairs = sorted(weights)
            G = ig.Graph(n=len(names), edges=[(idx[a], idx[b]) for a, b in pairs])
            w = [weights[p] for p in pairs]
            pr = G.pagerank(weights=w) if pairs else []
            deg = G.degree() if pairs else []
            scale = len(names)  # PageRank times network size: 1 is an average node, comparable across years
            for t in by_year[y]:
                vals = [pr[idx[n]] * scale for n in members[t] if n in idx]
                if prefix == "people":
                    rows[t]["people_pagerank_mean"] = float(np.mean(vals)) if vals else 0.0
                    rows[t]["people_pagerank_max"] = float(np.max(vals)) if vals else 0.0
                    rows[t]["people_degree_mean"] = float(np.mean([deg[idx[n]] if n in idx else 0 for n in members[t]])) if members[t] else 0.0
                else:
                    rows[t]["company_pagerank_mean"] = float(np.mean(vals)) if vals else 0.0
        for t in by_year[y]:  # then add this year's films
            for a, b in combinations(sorted(key_people[t]), 2):
                pw[(a, b)] += 1
            for a, b in combinations(sorted(companies[t]), 2):
                cw[(a, b)] += 1
        if y % 10 == 0:
            print(f"networks built through {y}: {len(pw):,} people links, {len(cw):,} company links", flush=True)

    cols = list(next(iter(rows.values())).keys())
    cols += [c for c in ("people_pagerank_mean", "people_pagerank_max", "people_degree_mean", "company_pagerank_mean")
             if c not in cols]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in sorted(rows):
            w.writerow({c: ("" if rows[t].get(c) is None else (round(rows[t][c], 6) if isinstance(rows[t][c], float) else rows[t][c]))
                        for c in cols})
    print(f"wrote {OUT}: {len(rows):,} movies")


if __name__ == "__main__":
    main()
