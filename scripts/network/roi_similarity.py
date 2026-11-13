"""Do films that share a director, writer, producer, lead actor, studio or
distributor have more similar returns than chance? Run from the data folder
after build_graph.py and franchise_flag.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/roi_similarity.py

Sample: the clean movies in the graph (the clean theatrical rule of
build_graph.py), with y = ln(gross / budget). Gross over budget isn't profit:
marketing is missing and cinemas keep part of the gross.

Groups: for each role, every person or company credited on at least two
sample films. Cast means the top three billed. A film sits in many groups at
once, one per credit.

Measures, per role:
- similarity: 1 - W / V, where W is the pooled within-group variance of y over
  every (group, film) credit and V the variance of y over the same credits. It
  is the share of variance that group membership accounts for, like an
  intraclass correlation. 0 means no more alike than random films.
- loo_r: Pearson correlation between a film's y and the mean y of the group's
  other films, over every credit. A second way to ask the same question.

Null: y shuffled across the sample films (default 1,000 times). That keeps
every group's size and every overlap between groups, and breaks only the link
between a film and its return. p is the one-sided share of shuffles at least
as similar as the data.

Confounds, as variants:
- raw: y as it is;
- adjusted: the residual of y after ln(budget), release decade and genres
  (OLS), since a director who makes cheap horror films in the 2010s will share
  returns for that reason alone;
- adjusted_no_franchise: adjusted, without films flagged in
  features/franchise.csv, since sequels share a director, a studio and a
  return.

The test is descriptive: it uses every film at once, so it says whether
returns cluster by group, not whether a group's past predicts a new film. For
that, see the *_track features in film_features.py, which use earlier films
only.

Writes features/roi_by_entity.csv (each group's films, mean y, and a mean
shrunk towards zero by empirical Bayes so that groups with two films don't top
the list by luck) and analysis/network/roi_similarity.json in the repo.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

NODES = "features/graph/nodes.csv"
EDGES = "features/graph/edges.csv"
FRANCHISE = "features/franchise.csv"
ENTITIES = "features/roi_by_entity.csv"
SUMMARY = Path(__file__).resolve().parents[2] / "analysis" / "network" / "roi_similarity.json"
ROLES = ["director", "writer", "producer", "cast", "composer", "cinematographer", "editor", "studio",
         "distributor"]
CAST_TOP = 3


def load():
    movies, labels = {}, {}
    with open(NODES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            labels[r["id"]] = r["label"]
            if r["type"] == "movie" and r["clean"] == "1" and r["multiple"]:
                movies[r["id"]] = {"y": math.log(float(r["multiple"])), "budget": float(r["budget_usd"]),
                                   "year": int(r["year"]), "genres": [g for g in r["genres"].split(",") if g]}
    members = defaultdict(set)  # (role, entity) -> movie ids
    with open(EDGES, newline="") as f:
        for r in csv.DictReader(f):
            if r["target"] not in movies:
                continue
            role = r["role"]
            if role == "actor":
                if not r["ordering"] or int(r["ordering"]) > CAST_TOP:
                    continue
                role = "cast"
            if role in ROLES:
                members[(role, r["source"])].add(r["target"])
    franchise = set()
    with open(FRANCHISE, newline="") as f:
        franchise = {r["tconst"] for r in csv.DictReader(f) if r["franchise"] == "1"}
    return movies, labels, members, franchise


def residuals(ids, movies):
    """y minus its OLS fit on ln(budget), release decade and genre dummies."""
    genres = sorted({g for i in ids for g in movies[i]["genres"]})
    decades = sorted({movies[i]["year"] // 10 for i in ids})[1:]  # first decade is the baseline
    x = np.array([[1.0, math.log(movies[i]["budget"])]
                  + [float(movies[i]["year"] // 10 == d) for d in decades]
                  + [float(g in movies[i]["genres"]) for g in genres] for i in ids])
    y = np.array([movies[i]["y"] for i in ids])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


class Role:
    """The credits of one role as (group, film) index pairs, for fast shuffles."""

    def __init__(self, groups, index):
        self.g = np.concatenate([np.full(len(m), k) for k, m in enumerate(groups)])
        self.m = np.concatenate([[index[i] for i in sorted(m)] for m in groups])
        self.n = np.bincount(self.g).astype(float)
        self.k = len(groups)

    def stats(self, y):
        v = y[self.m]
        s = np.bincount(self.g, weights=v, minlength=self.k)
        mean = s / self.n
        within = ((v - mean[self.g]) ** 2).sum() / (len(v) - self.k)
        similarity = 1 - within / v.var(ddof=1)
        others = (s[self.g] - v) / (self.n[self.g] - 1)  # mean of the group's other films
        loo = np.corrcoef(v, others)[0, 1]
        return similarity, loo


def test(ids, y, members, role, min_films, shuffles, rng):
    index = {i: k for k, i in enumerate(ids)}
    keys = [key for key, m in members.items() if key[0] == role and len(m & index.keys()) >= min_films]
    groups = [members[key] & index.keys() for key in keys]
    r = Role(groups, index)
    sim, loo = r.stats(y)
    null = np.array([r.stats(y[rng.permutation(len(y))]) for _ in range(shuffles)])
    return {
        "groups": r.k, "films": int(len(np.unique(r.m))), "credits": int(len(r.m)),
        "median_group_size": float(np.median(r.n)),
        "similarity": round(float(sim), 4),
        "similarity_null_mean": round(float(null[:, 0].mean()), 4),
        "similarity_null_95": round(float(np.quantile(null[:, 0], 0.95)), 4),
        "similarity_p": round(float((1 + (null[:, 0] >= sim).sum()) / (1 + shuffles)), 4),
        "loo_r": round(float(loo), 4),
        "loo_r_null_mean": round(float(null[:, 1].mean()), 4),
        "loo_r_null_95": round(float(np.quantile(null[:, 1], 0.95)), 4),
        "loo_r_p": round(float((1 + (null[:, 1] >= loo).sum()) / (1 + shuffles)), 4),
    }, keys, groups


def shrunk(groups, index, y, similarity):
    """Each group's mean y, shrunk towards zero by n * tau2 / (n * tau2 + sigma2)."""
    total = y.var(ddof=1)
    tau2, sigma2 = max(similarity, 0) * total, (1 - max(similarity, 0)) * total
    out = []
    for m in groups:
        v = y[[index[i] for i in m]]
        w = len(v) * tau2 / (len(v) * tau2 + sigma2) if tau2 > 0 else 0.0
        out.append((float(v.mean()), w * float(v.mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2805)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    movies, labels, members, franchise = load()
    ids = sorted(movies)
    no_fr = [i for i in ids if i not in franchise]
    variants = {
        "raw": (ids, np.array([movies[i]["y"] for i in ids])),
        "adjusted": (ids, residuals(ids, movies)),
        "adjusted_no_franchise": (no_fr, residuals(no_fr, movies)),
    }
    summary = {"films": len(ids), "franchise_films": len(set(ids) & franchise), "shuffles": args.shuffles,
               "seed": args.seed, "cast_top": CAST_TOP, "results": {}}
    rows = []
    for min_films in (2, 5):
        for name, (vids, y) in variants.items():
            for role in ROLES:
                res, keys, groups = test(vids, y, members, role, min_films, args.shuffles, rng)
                summary["results"].setdefault(f"min{min_films}", {}).setdefault(name, {})[role] = res
                print(f"min{min_films} {name:22s} {role:15s} groups {res['groups']:5d} "
                      f"sim {res['similarity']:+.3f} (null {res['similarity_null_mean']:+.3f}, "
                      f"p {res['similarity_p']:.3f})  loo_r {res['loo_r']:+.3f} (p {res['loo_r_p']:.3f})")
                if min_films == 2 and name == "adjusted":
                    index = {i: k for k, i in enumerate(vids)}
                    raw = {i: movies[i]["y"] for i in vids}
                    for key, m, (mean, shr) in zip(keys, groups, shrunk(groups, index, y, res["similarity"])):
                        rows.append([role, key[1], labels.get(key[1], key[1]), len(m),
                                     round(float(np.mean([raw[i] for i in m])), 4), round(mean, 4), round(shr, 4),
                                     sum(i in franchise for i in m)])
    with open(ENTITIES, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["role", "id", "name", "films", "mean_ln_multiple", "mean_adjusted", "shrunk_adjusted",
                    "franchise_films"])
        w.writerows(sorted(rows, key=lambda r: (r[0], -r[6])))
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {ENTITIES} ({len(rows)} groups) and {SUMMARY}")


if __name__ == "__main__":
    main()
