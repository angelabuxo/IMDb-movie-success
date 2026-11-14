"""When someone makes a film at a studio that is new to them, does their track
record come with them, or does the film return what that studio's films
return? Run from the data folder after build_graph.py and franchise_flag.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/movers.py

roi_similarity.py found that films sharing a person have similar returns and
that about half of that survives a shuffle within studio. This asks the same
question with moves, using only earlier films.

Sample and outcome are those of roi_similarity.py: the clean movies in the
graph, y = ln(gross / budget) minus its OLS fit on ln(budget), release decade
and genres. Gross over budget isn't profit.

Unit: one credit, a person in a role (director, writer, producer, top-three
cast) on a clean film that has a studio. For each credit, from films released
in earlier years only:
- person_track: the mean adjusted return of the person's earlier clean films
  (at least MIN_PERSON of them, any role);
- studio_track: the mean adjusted return of the earlier clean films of this
  film's studios, leaving out every film the person worked on (at least
  MIN_STUDIO of them), so the two tracks never share a film;
- mover: none of this film's studios appears on any of the person's earlier
  films in the graph. Otherwise the credit is a stayer.

For a mover, person_track was earned entirely at other studios, so its
coefficient is the part of a record that travels. For a stayer it also holds
whatever the studio contributed.

Tests, per role and for movers and stayers apart:
- OLS of the film's adjusted return on the two tracks, both standardised over
  the credits used, with a bootstrap interval that resamples people (a person
  has several credits). Null: returns shuffled among films released in the
  same year (default 1,000 times).
- matched: a person's record rests on a few films and a studio's on many, so
  the person's is noisier and its coefficient is pulled towards zero. Here
  both tracks use exactly MATCH films: the person's most recent ones, and a
  random draw of the studio's (averaged over DRAWS draws).

Variants: all films; no_franchise (films flagged in features/franchise.csv
dropped and residuals refitted), since a sequel keeps its people and return.

What it doesn't show: moves aren't random. A studio hires a director for a
project it has already chosen, so a mover's coefficient is not what would
happen if people were reassigned. "Studio" is the production company in the
Wikipedia infobox, and a film with several counts as a stay if any of them is
familiar. Histories cover films with 10,000+ IMDb votes only.

Writes analysis/network/movers.json in the repo.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from roi_similarity import CAST_TOP, EDGES, NODES, load, residuals

SUMMARY = Path(__file__).resolve().parents[2] / "analysis" / "network" / "movers.json"
ROLES = ["director", "writer", "producer", "cast"]
MIN_PERSON, MIN_STUDIO, MATCH, DRAWS = 2, 5, 3, 50


def graph_credits():
    """Over the whole graph: film years, each film's studios, each person's films, and the role credits."""
    year = {}
    with open(NODES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["type"] == "movie" and r["year"]:
                year[r["id"]] = int(r["year"])
    studios, films_of, studio_films, credits = defaultdict(set), defaultdict(set), defaultdict(set), set()
    with open(EDGES, newline="") as f:
        for r in csv.DictReader(f):
            t, role = r["target"], r["role"]
            if t not in year:
                continue
            if role == "studio":
                studios[t].add(r["source"])
                studio_films[r["source"]].add(t)
            elif role != "distributor":
                films_of[r["source"]].add(t)
                if role == "actor":
                    if not r["ordering"] or int(r["ordering"]) > CAST_TOP:
                        continue
                    role = "cast"
                if role in ROLES:
                    credits.add((role, r["source"], t))
    return year, studios, films_of, studio_films, sorted(credits)


def build(ids, resid, graph, rng):
    """One row per usable credit: role, person, film index, year, mover, and the two tracks (full and matched)."""
    year, studios, films_of, studio_films, credits = graph
    r_of, pos = dict(zip(ids, resid)), {i: k for k, i in enumerate(ids)}
    rows = []
    for role, person, film in credits:
        if film not in pos or not studios.get(film):
            continue
        y = year[film]
        earlier = [g for g in films_of[person] if year[g] < y]
        own = sorted((g for g in earlier if g in r_of), key=lambda g: (year[g], g))
        if len(own) < MIN_PERSON:
            continue
        mine = films_of[person]
        pool = sorted({g for s in studios[film] for g in studio_films[s]
                       if year[g] < y and g in r_of and g not in mine})
        if len(pool) < MIN_STUDIO:
            continue
        seen = set().union(*(studios.get(g, set()) for g in earlier))
        own_r, pool_r = np.array([r_of[g] for g in own]), np.array([r_of[g] for g in pool])
        matched_p = own_r[-MATCH:].mean() if len(own_r) >= MATCH else np.nan
        draws = [pool_r[rng.choice(len(pool_r), MATCH, replace=False)].mean() for _ in range(DRAWS)]
        rows.append((role, person, pos[film], y, not (studios[film] & seen), own_r.mean(), pool_r.mean(),
                     matched_p, draws, len(own_r), len(pool_r)))
    return rows


def within(strata, rng):
    perm = np.empty(len(strata), dtype=int)
    perm[np.argsort(strata, kind="stable")] = np.lexsort((rng.random(len(strata)), strata))
    return perm


def coefs(y, a, b):
    """Coefficients of standardised a and b in an OLS of y (in units of y's SD)."""
    def z(v):
        return (v - v.mean()) / v.std()
    x = np.column_stack([np.ones(len(y)), z(a), z(b)])
    return np.linalg.lstsq(x, z(y), rcond=None)[0][1:]


def test(rows, resid, film_years, k, rng):
    film = np.array([r[2] for r in rows])
    person = np.unique([r[1] for r in rows], return_inverse=True)[1]
    p, s = np.array([r[5] for r in rows]), np.array([r[6] for r in rows])
    y = resid[film]
    obs = coefs(y, p, s)
    null = np.array([coefs(resid[within(film_years, rng)][film], p, s) for _ in range(k)])
    by_person = defaultdict(list)
    for j, q in enumerate(person):
        by_person[q].append(j)
    groups = [np.array(v) for v in by_person.values()]
    boot = []
    for _ in range(k):
        pick = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        boot.append(coefs(y[pick], p[pick], s[pick]))
    boot = np.array(boot)
    out = {"credits": len(rows), "people": len(groups), "films": int(len(np.unique(film))),
           "person_films_median": float(np.median([r[9] for r in rows])),
           "studio_films_median": float(np.median([r[10] for r in rows])),
           "track_correlation": round(float(np.corrcoef(p, s)[0, 1]), 4)}
    for j, name in enumerate(("person", "studio")):
        out[name] = round(float(obs[j]), 4)
        out[f"{name}_ci"] = [round(float(q), 4) for q in np.quantile(boot[:, j], [0.025, 0.975])]
        out[f"{name}_null_95"] = round(float(np.quantile(null[:, j], 0.95)), 4)
        out[f"{name}_p"] = round(float((1 + (null[:, j] >= obs[j]).sum()) / (1 + k)), 4)
    m = np.array([not np.isnan(r[7]) for r in rows])
    if m.sum() > 50:
        draws = np.array([rows[j][8] for j in np.flatnonzero(m)])
        est = np.array([coefs(y[m], np.array([rows[j][7] for j in np.flatnonzero(m)]), draws[:, d])
                        for d in range(DRAWS)])
        out["matched"] = {"credits": int(m.sum()), "films_per_track": MATCH,
                          "person": round(float(est[:, 0].mean()), 4), "studio": round(float(est[:, 1].mean()), 4),
                          "studio_draw_sd": round(float(est[:, 1].std()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2805)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    movies, _, _, franchise = load()
    graph = graph_credits()
    ids = sorted(movies)
    summary = {"shuffles": args.shuffles, "seed": args.seed, "min_person_films": MIN_PERSON,
               "min_studio_films": MIN_STUDIO, "results": {}}
    for name, vids in (("all", ids), ("no_franchise", [i for i in ids if i not in franchise])):
        resid = residuals(vids, movies)
        years = np.array([movies[i]["year"] for i in vids])
        rows = build(vids, resid, graph, rng)
        res = summary["results"][name] = {}
        for role in ROLES + ["any"]:
            res[role] = {}
            for group, flag in (("movers", True), ("stayers", False)):
                sub = [r for r in rows if (role == "any" or r[0] == role) and r[4] == flag]
                r = res[role][group] = test(sub, resid, years, args.shuffles, rng)
                mt = r.get("matched", {})
                print(f"{name:13s} {role:9s} {group:8s} credits {r['credits']:6d} people {r['people']:5d} "
                      f"person {r['person']:+.3f} {r['person_ci']} studio {r['studio']:+.3f} {r['studio_ci']} "
                      f"null95 {r['person_null_95']:+.3f}/{r['studio_null_95']:+.3f} "
                      f"matched {mt.get('person', float('nan')):+.3f}/{mt.get('studio', float('nan')):+.3f}")
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
