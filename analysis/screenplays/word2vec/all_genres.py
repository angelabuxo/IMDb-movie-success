"""Hit vs flop in every genre, with the Word2Vec script vectors and course methods only (post hoc, exploratory).

The primary test (run_tests.py) covers Comedy and Action, with Drama as a check, and uses a cross-validated
logistic regression. This script asks the same question for every genre, so the screenplay result can sit next to
the genre result (Q1). It uses cosine similarity between script vectors and a label shuffle, nothing else.

Inputs
  the script vectors from vectors.py (window 5, TF-IDF-weighted mean), one per film, layer and seed.
  Every vectors_w5_s*.npz in W2V_VECTOR_DIR is used and the cosine is averaged over seeds, as run_tests.py
  averages its AUC over seeds 1 to 5.
  the hit / flop labels of genre_cells.label: top and bottom quarter of ln(gross/budget) within genre x era,
  middle half dropped, cells under 8 films dropped (DECISIONS.md D11 to D13)

Vectors are centred on the mean of all 2188 scripts before any cosine. Without that, every pair of scripts has a
cosine near 0.9, because all of them share the direction of common words.

Statistics, per genre and layer
  separation   mean cosine within hits and within flops, minus mean cosine between a hit and a flop.
               Above 0: hit scripts resemble each other, and flop scripts each other, more than they resemble
               scripts across the two groups.
  centroid     cosine distance between the hit centroid and the flop centroid.
  null         2000 shuffles of the hit / flop labels inside each era cell, so every shuffle keeps the era mix.
               p = (1 + shuffles at or above the observed value) / (1 + shuffles).
  holm         Holm correction over every genre x layer test of this script.

Genre check: the same separation statistic for "films of genre X" against "films of the final set without X",
with a shuffle of the genre label. That says how much genre the vector carries, on the same scale.

Outputs (numbers only) in results/:
  all_genres.csv        one row per genre x layer, hit vs flop
  all_genres_genre.csv  one row per genre x layer, genre vs the rest

Run:  IMDB_DATA_DIR=... python -I analysis/screenplays/word2vec/all_genres.py [--perms 2000]
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "genre_counts"))
from final_set import final_films  # noqa: E402
from genre_cells import era, label, pairs_for  # noqa: E402
from genre_counts import DATA  # noqa: E402

OUT = HERE / "results"
LAYERS = ("dialogue", "description")
VECTOR_DIR = Path(os.environ.get("W2V_VECTOR_DIR", DATA / "features" / "script_w2v"))
VECTORS = sorted(VECTOR_DIR.glob("vectors_w5_s*.npz"))   # every window-5 seed found
MIN_GROUP = 20   # smallest hit or flop group reported; under D14's 50 a row is marked small
SEED = 2805


def likely_franchise():
    """IMDb ids with a Wikidata 'follows' or 'franchise' statement, the flag of scripts_text/s10_franchise.py."""
    out = set()
    for name in ("wd_follows.tsv", "wd_franchise.tsv"):
        with open(DATA / "features" / name, newline="") as f:
            next(f)
            out |= {line.split("\t")[0].strip().strip('"') for line in f if line.strip()}
    return out


def load_vectors():
    """Per layer an array (seeds, films, 100) of centred, unit-length vectors, one slice per window-5 seed."""
    if not VECTORS:
        raise SystemExit(f"no vectors_w5_s*.npz in {VECTOR_DIR}")
    ids, out = None, {layer: [] for layer in LAYERS}
    for path in VECTORS:
        z = np.load(path)
        these = [str(i) for i in z["ids"]]
        if ids is not None and these != ids:
            raise SystemExit(f"{path.name}: film order differs from {VECTORS[0].name}")
        ids = these
        for layer in LAYERS:
            X = z[f"{layer}_tfidf_mean"].astype(float)
            X = X - np.nanmean(X, axis=0)
            out[layer].append(X / np.linalg.norm(X, axis=1, keepdims=True))
    return {i: k for k, i in enumerate(ids)}, {layer: np.stack(v) for layer, v in out.items()}


def separation(S, y):
    """Mean cosine within groups minus mean cosine between groups, for a 0/1 label y and similarity matrix S
    with zero diagonal."""
    a, b = y.astype(float), 1.0 - y
    na, nb = a.sum(), b.sum()
    within = (a @ S @ a + b @ S @ b) / (na * (na - 1) + nb * (nb - 1))
    between = (a @ S @ b) / (na * nb)
    return within - between


def centroid_distance(X, y):
    """Mean over seeds of the cosine distance between the hit and the flop centroid."""
    out = []
    for Xs in X:
        ca, cb = Xs[y == 1].mean(axis=0), Xs[y == 0].mean(axis=0)
        out.append(1.0 - ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))
    return float(np.mean(out))


def shuffle_within(y, strata, rng):
    out = y.copy()
    for idx in strata:
        out[idx] = rng.permutation(y[idx])
    return out


def test(X, y, strata, perms, rng):
    # Cosine averaged over seeds. The separation is linear in S, so this equals the mean of the per-seed
    # separations, with every seed seeing the same label shuffles.
    S = np.einsum("sik,sjk->ij", X, X) / X.shape[0]
    np.fill_diagonal(S, 0.0)
    obs = separation(S, y)
    null = np.array([separation(S, shuffle_within(y, strata, rng)) for _ in range(perms)])
    p = (1 + (null >= obs).sum()) / (1 + perms)
    z = (obs - null.mean()) / null.std()
    return obs, centroid_distance(X, y), null, p, z


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    index, vecs = load_vectors()
    films = [fl for fl in final_films() if fl["id"] in index]
    print(f"final set with a vector: {len(films)} films; seeds: {", ".join(p.stem for p in VECTORS)}")
    labelled, _ = label(pairs_for(films, "a"))

    franchise = likely_franchise()
    by_genre = defaultdict(list)
    for fl, g, e, lab in labelled:
        if lab != "middle":
            by_genre[g].append((fl["id"], e, fl["tier"], 1 if lab == "hit" else 0))
    genres = [g for g, items in sorted(by_genre.items(), key=lambda kv: -len(kv[1]))
              if min(sum(t[3] for t in items), sum(1 - t[3] for t in items)) >= MIN_GROUP]

    # primary: shuffle within era cells. era_x_tier: shuffle within era x vote tier, so the null keeps the tier
    # mix of hits and flops (D10). no_franchise: the primary test without sequels and named-franchise films (D26).
    checks = (("primary", lambda t: t[1], lambda t: True),
              ("era_x_tier", lambda t: (t[1], t[2]), lambda t: True),
              ("no_franchise", lambda t: t[1], lambda t: t[0] not in franchise))
    rows = []
    for check, stratum, keep in checks:
        for g in genres:
            items = [t for t in by_genre[g] if keep(t)]
            y = np.array([t[3] for t in items])
            n_hit, n_flop = int(y.sum()), int(len(y) - y.sum())
            rows_idx = np.array([index[t[0]] for t in items])
            cells = defaultdict(list)
            for k, t in enumerate(items):
                cells[stratum(t)].append(k)
            strata = [np.array(v) for v in cells.values()]
            for layer in LAYERS:
                obs, cdist, null, p, z = test(vecs[layer][:, rows_idx], y, strata, args.perms, rng)
                rows.append({"check": check, "genre": g, "layer": layer, "hits": n_hit, "flops": n_flop,
                             "small": min(n_hit, n_flop) < 50,
                             "separation": obs, "null_mean": null.mean(), "null_sd": null.std(),
                             "z": z, "p": p, "centroid_distance": cdist})
                print(f"{check:<13}{g:<10}{layer:<12}{n_hit:>4}/{n_flop:<4} sep {obs:+.4f}"
                      f"  null {null.mean():+.4f} ± {null.std():.4f}  z {z:+.2f}  p {p:.4f}")
        sel = [r for r in rows if r["check"] == check]
        for r, a in zip(sel, holm(np.array([r["p"] for r in sel]))):
            r["p_holm"] = a
    OUT.mkdir(exist_ok=True)
    with open(OUT / "all_genres.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    # Genre check: genre X against the rest of the final set, shuffled over the whole set within era.
    ids = np.array([index[fl["id"]] for fl in films])
    eras = np.array([era(fl["decade"]) for fl in films])
    strata = [np.where(eras == e)[0] for e in np.unique(eras)]
    genre_rows = []
    for g in genres:
        y = np.array([1 if g in fl["genres"] else 0 for fl in films])
        for layer in LAYERS:
            obs, cdist, null, p, z = test(vecs[layer][:, ids], y, strata, max(200, args.perms // 10), rng)
            genre_rows.append({"genre": g, "layer": layer, "in_genre": int(y.sum()), "rest": int(len(y) - y.sum()),
                               "separation": obs, "null_mean": null.mean(), "null_sd": null.std(), "z": z,
                               "p": p, "centroid_distance": cdist})
            print(f"genre {g:<10}{layer:<12} sep {obs:+.4f}  z {z:+.1f}  centroid {cdist:.3f}")
    with open(OUT / "all_genres_genre.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(genre_rows[0]))
        w.writeheader()
        for r in genre_rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
