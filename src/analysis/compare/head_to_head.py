"""Head to head on the common set, with one statistic and one null: do films that share a genre, share people or have
similar screenplays earn alike?

Course methods only: three film-film networks (week 1), return assortativity on each (week 3), and the same shuffle
null for all three (week 2); the script network uses cosine similarity of sentence embeddings.

  Films     in_common_set (usable money + a MovieSum script with both layers of 500+ tokens), the set Q1-Q3 share.
  Outcome   ln(gross / budget) minus the mean of its budget decile x era cell (q1_tmdb.net_return): the return a film
            makes beyond films of the same budget and era. Genre is deliberately not removed, so the three networks
            are compared on the same number.
  Networks  genre   two films are linked when they share at least one IMDb genre (genres with 100+ films in the set)
            people  two films are linked when they share a director, one of the first three billed actors, a producer
                    or a studio (Q2's links, network_by_genre.members)
            script  each film is linked to its K nearest films by cosine similarity of the screenplay's all-MiniLM-L6-v2
                    embedding (features/embeddings_hpc), links made undirected
  Statistic assortativity: the Pearson correlation of the outcome at the two ends of every link, counting each link in
            both directions (Newman's assortativity for a numeric node value).
  Splits    for the people and script networks, the links between films that share a genre and those that don't, so
            the part beyond genre shows.
  Null      the outcome shuffled within era bins, every network kept, SHUFFLES times; one-sided p.
What this measures: whether returns cluster along each kind of tie on the same films with the same number. It is
descriptive and uses all years at once, so it isn't a prediction.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.compare.head_to_head [--shuffles 1000] [--k 10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.compare.network_by_genre import members, pairs_of
from analysis.genre.q1_genre import DATA, SEED, load, prepare
from analysis.genre.q1_tmdb import net_return

OUT = Path(__file__).resolve().parent / "outputs" / "head_to_head"


def assortativity(y, a, b):
    """Newman's assortativity for a numeric value: Pearson over links counted in both directions."""
    if len(a) < 3:
        return np.nan
    return float(np.corrcoef(np.concatenate([y[a], y[b]]), np.concatenate([y[b], y[a]]))[0, 1])


def genre_links(M):
    share = (M @ M.T) > 0
    a, b = np.nonzero(np.triu(share, 1))
    return a, b


def people_links(f):
    ids = set(f["imdb_id"])
    index = {t: i for i, t in enumerate(f["imdb_id"])}
    mem = members(ids)
    pairs = set()
    for role in mem:
        pairs.update(pairs_of(mem[role], ids))
    P = sorted(pairs)
    return np.array([index[i] for i, _ in P]), np.array([index[j] for _, j in P])


def script_links(f, k):
    V = np.load(DATA / "features" / "embeddings_hpc" / "script_vectors.npy")
    ids = pd.read_csv(DATA / "features" / "embeddings_hpc" / "ids.csv")["tconst"]
    pos = {t: i for i, t in enumerate(ids)}
    r = np.array([pos.get(t, -1) for t in f["imdb_id"]])
    assert (r >= 0).all(), "a common-set film has no script vector"
    X = V[r].astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    pairs = {(min(i, j), max(i, j)) for i in range(len(X)) for j in nn[i]}
    P = sorted(pairs)
    return np.array([i for i, _ in P]), np.array([j for _, j in P])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "in_common_set")
    y = net_return(f)
    eras = f["era_bin"].astype(str).to_numpy()
    era_pos = [np.flatnonzero(eras == e) for e in pd.unique(eras)]
    rng = np.random.default_rng(SEED)
    nulls = []
    for _ in range(a.shuffles):
        z = y.copy()
        for p in era_pos:
            z[p] = y[rng.permutation(p)]
        nulls.append(z)
    shares_genre = (M @ M.T) > 0
    nets = {"genre": genre_links(M), "people": people_links(f), "script": script_links(f, a.k)}
    rows = []
    for name, (u, v) in nets.items():
        same = shares_genre[u, v]
        splits = {"all links": np.ones(len(u), bool)}
        if name != "genre":
            splits |= {"links within a shared genre": same, "links with no shared genre": ~same}
        for split, m in splits.items():
            obs = assortativity(y, u[m], v[m])
            nul = np.array([assortativity(z, u[m], v[m]) for z in nulls])
            rows.append({"network": name, "links": split, "n_links": int(m.sum()),
                         "mean_degree": round(2 * m.sum() / len(y), 1), "assortativity": round(obs, 4),
                         "null_mean": round(float(nul.mean()), 4), "null_95": round(float(np.quantile(nul, .95)), 4),
                         "p": round(float((1 + (nul >= obs).sum()) / (1 + a.shuffles)), 4),
                         "share_links_sharing_genre": round(float(same.mean()), 3)})
        print(pd.DataFrame(rows[-len(splits):]).to_string(index=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "assortativity.csv", index=False)
    summary = {"films": len(f), "genres": len(keep), "k_script": a.k, "shuffles": a.shuffles, "seed": SEED,
               "outcome": "ln(gross/budget) net of budget decile x era mean"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
