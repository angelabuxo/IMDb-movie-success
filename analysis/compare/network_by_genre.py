"""Does the people-network result (Q2) differ between genres, and is it carried by films of the same genre?

Rebuilds Q2's outcome assortativity exactly as scripts/network/q2_network.py on the Q2-Niklas branch does it, and
checks the rebuild against that branch's q2_network.json, then splits it.

  Films     in_common_set (1,598, Q2's primary set) and usable_money (all films with clean money, a check).
  Outcome   ln(gross / budget) minus an OLS fit on ln(budget), era-bin dummies and every IMDb genre tag (Q2's residual).
  Links     two films are linked when they share a director, one of the first three billed actors, a producer or a
            studio (features/graph/edges.csv), one graph per role, as in Q2.
  Splits    same genre   pairs whose two films share at least one IMDb genre, against pairs that share none
            per genre    pairs where both films carry the genre (genres with at least MIN_PAIRS such pairs)
  Statistic Pearson correlation of the residual returns across the pairs (Q2's assortativity, pairs ordered by id).
  Null      residual returns shuffled within era bins, the graph kept (as in Q2), SHUFFLES times; one-sided p for each
            split. For same against different genre, the difference of the two correlations against the same shuffles,
            two-sided. Per genre, Holm over the genres of each role.
What this measures: whether returns cluster along shared credits inside a genre and across genres. A same-genre
excess fits genre specialists; it doesn't separate specialists from a genre-specific team effect.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.compare.network_by_genre [--shuffles 1000]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.genre.q1_genre import DATA, SEED, holm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs" / "network_by_genre"
ROLES = ("director", "actor", "producer", "studio")
CAST_TOP = 3
ERA_ORDER = ("before 1980", "1980s", "1990s", "2000s", "2010s", "2020s")
MIN_PAIRS = 200
SAMPLES = {"in_common_set": "in_common_set", "usable_money": "usable_money"}


def residualize(frame):
    """Q2's residual: ln(gross/budget) minus the OLS fit on ln(budget), era and every genre tag."""
    genres = sorted({g for v in frame["genres"].fillna("") for g in v.split(",") if g})
    eras = [e for e in ERA_ORDER if e in set(frame["era_bin"].dropna())]
    x = np.column_stack([np.ones(len(frame)), np.log(frame["budget"].astype(float).to_numpy()),
                         *[frame["era_bin"].eq(e).astype(float).to_numpy() for e in eras[1:]],
                         *[frame["genres"].fillna("").apply(lambda v, g=g: float(g in set(v.split(",")))).to_numpy() for g in genres]])
    y = frame["ln_gross_over_budget"].astype(float).to_numpy()
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def members(ids):
    m = {r: defaultdict(set) for r in ROLES}
    with open(DATA / "features" / "graph" / "edges.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["target"] not in ids or row["role"] not in m:
                continue
            if row["role"] == "actor":
                try:
                    if int(row["ordering"]) > CAST_TOP:
                        continue
                except (TypeError, ValueError):
                    continue
            m[row["role"]][row["source"]].add(row["target"])
    return m


def pairs_of(mem, ids):
    pairs = set()
    for films in mem.values():
        pairs.update(combinations(sorted(films & ids), 2))
    return sorted(pairs)


def corr(y, a, b):
    if len(a) < 3:
        return np.nan
    return float(np.corrcoef(y[a], y[b])[0, 1])


def shuffled(y, era_pos, rng):
    out = y.copy()
    for pos in era_pos:
        out[pos] = y[rng.permutation(pos)]
    return out


def run_sample(frame, mem, shuffles, seed, reference):
    ids = set(frame["imdb_id"])
    index = {t: i for i, t in enumerate(frame["imdb_id"])}
    y = residualize(frame)
    eras = frame["era_bin"].astype(str).to_numpy()
    era_pos = [np.flatnonzero(eras == e) for e in pd.unique(eras)]
    gsets = [set(v.split(",")) - {""} for v in frame["genres"].fillna("")]
    all_genres = sorted(set().union(*gsets))
    rng = np.random.default_rng(seed)
    nulls = [shuffled(y, era_pos, rng) for _ in range(shuffles)]
    rows = []
    for role in ROLES:
        P = pairs_of(mem[role], ids)
        a = np.array([index[i] for i, _ in P])
        b = np.array([index[j] for _, j in P])
        same = np.array([bool(gsets[i] & gsets[j]) for i, j in zip(a, b)])
        splits = {"all": np.ones(len(a), bool), "same genre": same, "different genre": ~same}
        for g in all_genres:
            m = np.array([g in gsets[i] and g in gsets[j] for i, j in zip(a, b)])
            if m.sum() >= MIN_PAIRS:
                splits[f"genre: {g}"] = m
        obs = {k: corr(y, a[m], b[m]) for k, m in splits.items()}
        nul = {k: np.array([corr(n, a[m], b[m]) for n in nulls]) for k, m in splits.items()}
        if reference is not None:
            ref = reference[role]["assortativity"]
            assert round(obs["all"], 4) == ref, f"{role}: rebuild gives {obs['all']:.4f}, Q2 gives {ref}"
        diff_obs = obs["same genre"] - obs["different genre"]
        diff_null = nul["same genre"] - nul["different genre"]
        for k, m in splits.items():
            rows.append({"role": role, "split": k, "pairs": int(m.sum()), "assortativity": round(obs[k], 4),
                         "null_mean": round(float(np.nanmean(nul[k])), 4), "null_95": round(float(np.nanquantile(nul[k], .95)), 4),
                         "p": round(float((1 + np.sum(nul[k] >= obs[k])) / (1 + shuffles)), 4)})
        rows.append({"role": role, "split": "same minus different", "pairs": int(len(a)), "assortativity": round(diff_obs, 4),
                     "null_mean": round(float(np.nanmean(diff_null)), 4), "null_95": round(float(np.nanquantile(diff_null, .95)), 4),
                     "p": round(float((1 + np.sum(np.abs(diff_null - np.nanmean(diff_null)) >= abs(diff_obs - np.nanmean(diff_null)))) / (1 + shuffles)), 4)})
    d = pd.DataFrame(rows)
    d["p_holm_within_role"] = np.nan
    for role in ROLES:
        g = d[(d.role == role) & d.split.str.startswith("genre: ")]
        d.loc[g.index, "p_holm_within_role"] = np.round(holm(g.p.to_numpy()), 4)
    return d


def reference_json():
    for ref in ("public/Q2-Niklas", "origin/Q2-Niklas", "Q2-Niklas"):
        r = subprocess.run(["git", "show", f"{ref}:analysis/network/q2_network.json"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            return json.loads(r.stdout)
    local = ROOT / "analysis" / "network" / "q2_network.json"     # a copy, where the repo isn't a git checkout
    return json.loads(local.read_text()) if local.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=1000)
    a = ap.parse_args()
    films = pd.read_csv(DATA / "features" / "film_analysis_set.csv")
    ref = reference_json()
    OUT.mkdir(parents=True, exist_ok=True)
    out, summary = [], {"shuffles": a.shuffles, "seed": SEED, "min_pairs": MIN_PAIRS, "reference_checked": ref is not None}
    for name, flag in SAMPLES.items():
        frame = films[films[flag].fillna(False).astype(bool)].reset_index(drop=True)
        mem = members(set(frame["imdb_id"]))
        refs = ref["samples"][f"{name}_adjusted"] if ref else None
        d = run_sample(frame, mem, a.shuffles, SEED, refs)
        d.insert(0, "sample", name)
        out.append(d)
        summary[name] = {"films": len(frame)}
        print(d.to_string(index=False), flush=True)
    pd.concat(out).to_csv(OUT / "assortativity_by_genre.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
