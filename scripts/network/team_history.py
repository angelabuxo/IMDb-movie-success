"""Do films made by people who have worked together before return more? Run
from the data folder after build_graph.py and franchise_flag.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/team_history.py

Sample and outcome are those of roi_similarity.py: the clean movies in the
graph, y = ln(gross / budget) minus its OLS fit on ln(budget), release decade
and genres. Gross over budget isn't profit.

A film's team is its key people: directors, writers, producers, top-three
cast, composers, cinematographers and editors. Everything about the team is
built from films released in earlier years only, over all 12,673 films in the
graph, so the film's own result can't leak in:
- repeat_share: the share of pairs of team members who share an earlier film;
- any_repeat: 1 when at least one pair does (a second, cruder measure);
- newcomer_share: the share of the team with no earlier film in the graph;
- experience: the team's mean number of earlier films;
- own_track: the mean adjusted return of the team's earlier clean films.

Tests:
- Spearman correlation of each measure with the adjusted return, and the same
  net of own_track, experience and team size (ranks, residualised by OLS).
  Null: returns shuffled among films released in the same year (default 1,000
  times), which keeps every year's level and breaks only who made the film.
- The same by era (to 1989, 1990 to 2004, 2005 on), with bootstrap intervals.
- Out of time: OLS fitted on films to 2010 and scored on 2011 to 2017, with
  own_track, experience and team size, then adding the two shares. The gain
  in R2 gets a bootstrap interval over the test films.
- Mean adjusted return by level of repeat_share, for a figure.

Variants: all films; no_franchise (films flagged in features/franchise.csv
dropped and residuals refitted), since a sequel reunites its team and shares
its return.

What it doesn't show: "earlier film" means an earlier film with 10,000+ IMDb
votes, so a newcomer may have a long career in smaller films. IMDb's
principals file lists about ten people a title, so teams are partial. The
adjusted return is fitted on every year at once. Nothing here is causal:
studios may reunite teams for projects they already expect to do well.

Writes analysis/network/team_history.json in the repo.
"""

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from roi_similarity import CAST_TOP, EDGES, NODES, load, residuals

SUMMARY = Path(__file__).resolve().parents[2] / "analysis" / "network" / "team_history.json"
PEOPLE = {"director", "writer", "producer", "actor", "composer", "cinematographer", "editor"}
ERAS = [("to_1989", 0, 1989), ("1990_2004", 1990, 2004), ("2005_on", 2005, 9999)]
TRAIN_TO, TEST_TO = 2010, 2017


def graph_history():
    """Every film's year and key people, and every person's films, over the whole graph."""
    year = {}
    with open(NODES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if r["type"] == "movie" and r["year"]:
                year[r["id"]] = int(r["year"])
    team, films = defaultdict(set), defaultdict(set)
    with open(EDGES, newline="") as f:
        for r in csv.DictReader(f):
            if r["target"] not in year or r["role"] not in PEOPLE:
                continue
            if r["role"] == "actor" and (not r["ordering"] or int(r["ordering"]) > CAST_TOP):
                continue
            team[r["target"]].add(r["source"])
            films[r["source"]].add(r["target"])
    return year, team, films


def features(ids, resid, year, team, films):
    """One row per film: repeat_share, any_repeat, newcomer_share, experience, own_track, size (nan when undefined)."""
    r_of = dict(zip(ids, resid))
    out = np.full((len(ids), 6), np.nan)
    for k, i in enumerate(ids):
        people = sorted(team.get(i, ()))
        if not people:
            continue
        prior = [{g for g in films[p] if year[g] < year[i]} for p in people]
        out[k, 2] = np.mean([not s for s in prior])
        out[k, 3] = np.mean([len(s) for s in prior])
        out[k, 5] = len(people)
        if len(people) > 1:
            shared = [bool(prior[a] & prior[b]) for a, b in combinations(range(len(people)), 2)]
            out[k, 0], out[k, 1] = np.mean(shared), float(any(shared))
        earlier = [r_of[g] for g in set().union(*prior) if g in r_of]
        if earlier:
            out[k, 4] = np.mean(earlier)
    return out


def within(strata, rng):
    """A permutation of positions that only moves films within their stratum."""
    perm = np.empty(len(strata), dtype=int)
    perm[np.argsort(strata, kind="stable")] = np.lexsort((rng.random(len(strata)), strata))
    return perm


def rank_resid(v, controls):
    x = np.column_stack([np.ones(len(v))] + [rankdata(c) for c in controls])
    r = rankdata(v)
    return r - x @ np.linalg.lstsq(x, r, rcond=None)[0]


def corr_test(v, y, years, controls, k, rng):
    """Spearman of v with y, and net of the controls, each against a within-year shuffle of y."""
    rv, rvc = rankdata(v), rank_resid(v, controls)

    def stats(yy):
        return np.corrcoef(rv, rankdata(yy))[0, 1], np.corrcoef(rvc, rank_resid(yy, controls))[0, 1]

    obs = stats(y)
    null = np.array([stats(y[within(years, rng)]) for _ in range(k)])
    out = {"films": len(v)}
    for j, name in enumerate(("rho", "rho_partial")):
        out[name] = round(float(obs[j]), 4)
        out[f"{name}_null_95"] = round(float(np.quantile(null[:, j], 0.95)), 4)
        out[f"{name}_null_05"] = round(float(np.quantile(null[:, j], 0.05)), 4)
        out[f"{name}_p_two_sided"] = round(float((1 + (np.abs(null[:, j]) >= abs(obs[j])).sum()) / (1 + k)), 4)
    return out


def boot_rho(v, y, k, rng):
    rv, ry = rankdata(v), rankdata(y)
    n = len(v)
    draws = [np.corrcoef(rv[s], ry[s])[0, 1] for s in (rng.integers(0, n, n) for _ in range(k))]
    return [round(float(q), 4) for q in np.quantile(draws, [0.025, 0.975])]


def out_of_time(feat, y, years, k, rng):
    """R2 on 2011-2017 of OLS fitted to 2010: track, experience and size, then with the two shares added."""
    track = np.where(np.isnan(feat[:, 4]), 0.0, feat[:, 4])
    base = [track, np.isnan(feat[:, 4]).astype(float), np.log1p(feat[:, 3]), feat[:, 5]]
    blocks = {"track_experience_size": base, "plus_repeat_share": base + [feat[:, 0]],
              "plus_repeat_and_newcomer": base + [feat[:, 0], feat[:, 2]]}
    ok = ~np.isnan(feat[:, 0])
    tr, te = ok & (years <= TRAIN_TO), ok & (years > TRAIN_TO) & (years <= TEST_TO)
    err = {}
    for name, cols in blocks.items():
        x = np.column_stack([np.ones(len(y))] + cols)
        beta = np.linalg.lstsq(x[tr], y[tr], rcond=None)[0]
        err[name] = (y[te] - x[te] @ beta) ** 2
    tot = (y[te] - y[tr].mean()) ** 2
    out = {"train_films": int(tr.sum()), "test_films": int(te.sum()),
           "r2": {name: round(float(1 - e.sum() / tot.sum()), 4) for name, e in err.items()}}
    n = len(tot)
    samples = [rng.integers(0, n, n) for _ in range(k)]
    for name in ("plus_repeat_share", "plus_repeat_and_newcomer"):
        gain = [(err["track_experience_size"][s].sum() - err[name][s].sum()) / tot[s].sum() for s in samples]
        out[f"gain_{name}"] = round(out["r2"][name] - out["r2"]["track_experience_size"], 4)
        out[f"gain_{name}_ci"] = [round(float(q), 4) for q in np.quantile(gain, [0.025, 0.975])]
    return out


def levels(share, y):
    """Mean adjusted return for films with no repeat pair, then for thirds of the films that have one."""
    cuts = np.quantile(share[share > 0], [1 / 3, 2 / 3])
    masks = [("none", share == 0), ("low", (share > 0) & (share <= cuts[0])),
             ("middle", (share > cuts[0]) & (share <= cuts[1])), ("high", share > cuts[1])]
    return [{"level": name, "films": int(m.sum()), "repeat_share_mean": round(float(share[m].mean()), 4),
             "return_mean": round(float(y[m].mean()), 4),
             "return_se": round(float(y[m].std(ddof=1) / np.sqrt(m.sum())), 4)} for name, m in masks]


def run(ids, movies, graph, k, rng):
    year, team, films = graph
    y = residuals(ids, movies)
    years = np.array([movies[i]["year"] for i in ids])
    feat = features(ids, y, year, team, films)
    ok = ~np.isnan(feat[:, 0])
    f, yy, yrs = feat[ok], y[ok], years[ok]
    track = np.where(np.isnan(f[:, 4]), 0.0, f[:, 4])
    controls = [track, f[:, 3], f[:, 5]]
    out = {"films": len(ids), "films_with_team_of_two": int(ok.sum()),
           "share_with_any_repeat": round(float(f[:, 1].mean()), 4), "measures": {}, "eras": {}}
    for name, col in (("repeat_share", 0), ("any_repeat", 1), ("newcomer_share", 2)):
        out["measures"][name] = corr_test(f[:, col], yy, yrs, controls, k, rng)
    for name, lo, hi in ERAS:
        m = (yrs >= lo) & (yrs <= hi)
        res = corr_test(f[m, 0], yy[m], yrs[m], [c[m] for c in controls], k, rng)
        res["rho_ci"] = boot_rho(f[m, 0], yy[m], k, rng)
        res.update(team_size_mean=round(float(f[m, 5].mean()), 2), repeat_share_mean=round(float(f[m, 0].mean()), 4),
                   share_with_any_repeat=round(float(f[m, 1].mean()), 4))
        out["eras"][name] = res
    out["out_of_time"] = out_of_time(feat, y, years, k, rng)
    out["levels"] = levels(f[:, 0], yy)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2805)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    movies, _, _, franchise = load()
    graph = graph_history()
    ids = sorted(movies)
    summary = {"shuffles": args.shuffles, "seed": args.seed, "train_to": TRAIN_TO, "test_to": TEST_TO, "results": {}}
    for name, vids in (("all", ids), ("no_franchise", [i for i in ids if i not in franchise])):
        res = summary["results"][name] = run(vids, movies, graph, args.shuffles, rng)
        for key, r in res["measures"].items():
            print(f"{name:13s} {key:15s} rho {r['rho']:+.3f} (null 5-95% {r['rho_null_05']:+.3f} to {r['rho_null_95']:+.3f}) "
                  f"partial {r['rho_partial']:+.3f} (p {r['rho_partial_p_two_sided']:.3f})")
        for key, r in res["eras"].items():
            print(f"{name:13s} era {key:11s} n {r['films']:5d} rho {r['rho']:+.3f} ci {r['rho_ci']} partial {r['rho_partial']:+.3f} "
                  f"team {r['team_size_mean']} any repeat {r['share_with_any_repeat']}")
        print(f"{name:13s} out of time {res['out_of_time']}")
        print(f"{name:13s} levels {[(l['level'], l['films'], l['return_mean']) for l in res['levels']]}")
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
