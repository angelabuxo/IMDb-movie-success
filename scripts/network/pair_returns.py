"""How far does return similarity reach in the movie graph, does it grow with
the number of shared people, and do similar scripts explain it? Run from the
data folder after build_graph.py and franchise_flag.py (part C also needs the
screenplay embeddings in features/embeddings_hpc/):

    cd data/raw && ../../.venv/bin/python ../../scripts/network/pair_returns.py

Sample and outcome are those of roi_similarity.py: the clean movies in the
graph, y = ln(gross / budget) minus its OLS fit on ln(budget), release decade
and genres, then standardised. Gross over budget isn't profit.

The network is the film-film projection of the movie graph over key people
(directors, writers, producers, top-three cast, composers, cinematographers,
editors). Two films are linked when they share at least one of them, and the
link's weight is the number of people they share. Companies are left out of
the links and used as a control instead.

Measure: the pair correlation of a set of film pairs, mean(z_i * z_j). It is
the Pearson correlation between the two films of a pair when every pair is
counted in both orders. 0 means the pairs are no more alike than random films.

A. Distance. Pairs at shortest-path distance 1, 2, 3 and 4 in the projection,
   and pairs further apart or unconnected.
B. Dose. Linked pairs by the number of people they share: 1, 2, 3-4, 5+.
   Also linked pairs with no studio or distributor in common, and unlinked
   pairs that share a company.
C. Scripts. On films with a screenplay embedding (MovieSum, mean
   all-MiniLM-L6-v2 passage vector), the cosine similarity of two scripts
   against their return similarity: linked pairs, the same number of unlinked
   pairs with the most similar scripts, and unlinked pairs by decile of script
   similarity. Also whether linked films have more similar scripts than
   unlinked ones in the same first genre.

Nulls (default 1,000 shuffles, which keep the graph and break only the link
between a film and its return):
- all: z shuffled across every film;
- year: z shuffled only among films released in the same year, since linked
  films are close in time and a year can be good for everyone;
- studio: z shuffled only among films of the same main studio (A and B).
For the script similarity of linked films, the script vectors are shuffled
among films of the same first genre instead.

Variants: all films; no_franchise (films flagged in features/franchise.csv
dropped, residuals refitted); rank (returns mapped to normal scores first, so
no single film dominates).

What it doesn't show: it is descriptive and uses every year at once, so it is
about clustering, not prediction. Cosine similarity of mean passage vectors
measures shared topic and vocabulary, not quality or structure. The franchise
flag is rough, so some sequel pairs remain in no_franchise.

Writes analysis/network/pair_returns.json in the repo.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from roi_similarity import load, main_studio, normal_scores, residuals

EMB_IDS = "features/embeddings_hpc/ids.csv"
EMB_VEC = "features/embeddings_hpc/script_vectors.npy"
SUMMARY = Path(__file__).resolve().parents[2] / "analysis" / "network" / "pair_returns.json"
PEOPLE = ["director", "writer", "producer", "cast", "composer", "cinematographer", "editor"]
COMPANIES = ["studio", "distributor"]
DOSES = [("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5+", 5, 10 ** 6)]
MAX_DISTANCE = 4


def projection(ids, members, roles):
    """Symmetric film-by-film matrix of the number of distinct entities two films share (zero diagonal)."""
    index = {i: k for k, i in enumerate(ids)}
    ents, rows, cols = {}, [], []
    for (role, e), films in members.items():
        if role in roles:
            for i in films:
                if i in index:
                    rows.append(index[i])
                    cols.append(ents.setdefault(e, len(ents)))
    b = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(ids), len(ents)))
    b.data[:] = 1  # a person with two roles on a film counts once
    a = (b @ b.T).tolil()
    a.setdiag(0)
    a = a.tocsr()
    a.eliminate_zeros()
    return a


def binary(m):
    m = m.copy().tocsr()
    m.data[:] = 1
    return m


def minus(a, b):
    """Entries of binary a that are not in binary b."""
    out = (a - a.multiply(b)).tocsr()
    out.eliminate_zeros()
    return out


def rings(a1, max_distance):
    """Binary matrices of the film pairs at each shortest-path distance, then everything further or unconnected."""
    n = a1.shape[0]
    eye = sp.identity(n, dtype=np.float32, format="csr")
    reached, out = binary(a1 + eye), {1: a1}
    for d in range(2, max_distance + 1):
        nxt = binary(reached @ a1)
        out[d] = minus(nxt, reached)
        reached = binary(reached + nxt)
    return out, reached


def shuffles(z, strata, k, rng):
    """k shuffled copies of z as columns, permuted across all films or within strata."""
    out = np.empty((len(z), k))
    if strata is None:
        for j in range(k):
            out[:, j] = z[rng.permutation(len(z))]
        return out
    base = np.argsort(strata, kind="stable")
    for j in range(k):
        out[base, j] = z[np.lexsort((rng.random(len(z)), strata))]
    return out


def pair_corr(m, z):
    """mean(z_i * z_j) over the pairs of symmetric binary m; z may hold one column per shuffle."""
    return (z * (m @ z)).sum(axis=0) / m.nnz


def far_corr(reached, z):
    """The same over every pair not in `reached` (which includes the diagonal), without building that matrix."""
    n = len(z)
    total = z.sum(axis=0) ** 2  # sum over all ordered pairs including i = i
    return (total - (z * (reached @ z)).sum(axis=0)) / (n * n - reached.nnz)


def summarise(obs, nulls, pairs):
    out = {"pairs": int(pairs), "corr": round(float(obs), 4)}
    for name, null in nulls.items():
        out[f"null_{name}_mean"] = round(float(null.mean()), 4)
        out[f"null_{name}_95"] = round(float(np.quantile(null, 0.95)), 4)
        out[f"p_{name}"] = round(float((1 + (null >= obs).sum()) / (1 + len(null))), 4)
    return out


def standardise(y):
    return (y - y.mean()) / y.std()


def graph_tests(ids, y, movies, members, k, rng):
    z = standardise(y)
    people, comp = projection(ids, members, PEOPLE), binary(projection(ids, members, COMPANIES))
    a1 = binary(people)
    year = np.array([movies[i]["year"] for i in ids])
    sets = {"all": shuffles(z, None, k, rng), "year": shuffles(z, year, k, rng),
            "studio": shuffles(z, main_studio(ids, members), k, rng)}
    zc = z[:, None]

    def run(m):
        return summarise(pair_corr(m, zc)[0], {name: pair_corr(m, s) for name, s in sets.items()}, m.nnz // 2)

    by_distance, reached = rings(a1, MAX_DISTANCE)
    out = {"films": len(ids), "linked_films": int((a1.getnnz(axis=1) > 0).sum()),
           "distance": {str(d): run(m) for d, m in by_distance.items()}, "dose": {}, "company": {}}
    n = len(ids)
    out["distance"][f">{MAX_DISTANCE}"] = summarise(
        far_corr(reached, zc)[0], {name: far_corr(reached, s) for name, s in sets.items()},
        (n * n - reached.nnz) // 2)
    for label, lo, hi in DOSES:
        m = people.copy()
        m.data = ((m.data >= lo) & (m.data <= hi)).astype(np.float32)
        m.eliminate_zeros()
        out["dose"][label] = run(m)
    out["company"]["linked_no_shared_company"] = run(minus(a1, comp))
    out["company"]["linked_shared_company"] = run(a1.multiply(comp).tocsr())
    out["company"]["unlinked_shared_company"] = run(minus(comp, a1))
    return out


def script_tests(ids, y, movies, members, k, rng):
    with open(EMB_IDS, newline="", encoding="utf8") as f:
        emb_ids = [r["tconst"] for r in csv.DictReader(f)]
    vec = np.load(EMB_VEC)
    row = {t: j for j, t in enumerate(emb_ids)}
    pos = {i: j for j, i in enumerate(ids)}
    sub = [i for i in ids if i in row]
    v = vec[[row[i] for i in sub]].astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    z = standardise(y[[pos[i] for i in sub]])
    n = len(sub)
    linked = projection(sub, members, PEOPLE).toarray() > 0
    iu = np.triu_indices(n, 1)
    link, cos = linked[iu], (v @ v.T)[iu]
    genre = np.array([movies[i]["genres"][0] if movies[i]["genres"] else "" for i in sub])
    same_genre = (genre[:, None] == genre[None, :])[iu]
    year = np.array([movies[i]["year"] for i in sub])
    nulls = {"all": shuffles(z, None, k, rng), "year": shuffles(z, year, k, rng)}

    def run(mask):
        a, b = iu[0][mask], iu[1][mask]
        obs = float(np.mean(z[a] * z[b]))
        return summarise(obs, {name: (s[a] * s[b]).mean(axis=0) for name, s in nulls.items()}, mask.sum())

    out = {"films": n, "model": "all-MiniLM-L6-v2, mean passage vector", "pairs": {}, "unlinked_by_cosine_decile": []}
    order = np.argsort(-np.where(link, -np.inf, cos))  # unlinked pairs, most similar scripts first
    top = np.zeros(len(cos), dtype=bool)
    top[order[:link.sum()]] = True
    median = np.median(cos[link])
    out["pairs"]["linked"] = run(link)
    out["pairs"]["unlinked_most_similar_scripts"] = run(top)
    out["pairs"]["unlinked_most_similar_scripts"]["min_cosine"] = round(float(cos[top].min()), 4)
    out["pairs"]["unlinked_rest"] = run(~link & ~top)
    out["pairs"]["linked_similar_scripts"] = run(link & (cos >= median))
    out["pairs"]["linked_dissimilar_scripts"] = run(link & (cos < median))
    edges = np.quantile(cos[~link], np.linspace(0, 1, 11))
    for d in range(10):
        mask = ~link & (cos >= edges[d]) & ((cos < edges[d + 1]) if d < 9 else (cos <= edges[d + 1]))
        res = run(mask)
        res["cosine_mean"] = round(float(cos[mask].mean()), 4)
        out["unlinked_by_cosine_decile"].append(res)

    # Do linked films have more similar scripts than unlinked films of the same first genre?
    def gap(vv):
        c = (vv @ vv.T)[iu]
        return c[link & same_genre].mean() - c[~link & same_genre].mean()

    codes = np.unique(genre, return_inverse=True)[1]
    base = np.argsort(codes, kind="stable")
    null = []
    for _ in range(min(k, 200)):
        perm = np.empty(n, dtype=int)
        perm[base] = np.lexsort((rng.random(n), codes))
        null.append(gap(v[perm]))
    null = np.array(null)
    out["script_similarity"] = {
        "cosine_linked": round(float(cos[link].mean()), 4), "cosine_unlinked": round(float(cos[~link].mean()), 4),
        "cosine_linked_same_genre": round(float(cos[link & same_genre].mean()), 4),
        "cosine_unlinked_same_genre": round(float(cos[~link & same_genre].mean()), 4),
        "same_genre_gap": round(float(gap(v)), 4), "same_genre_gap_null_95": round(float(np.quantile(null, 0.95)), 4),
        "same_genre_gap_p": round(float((1 + (null >= gap(v)).sum()) / (1 + len(null))), 4),
        "cosine_sd_all_pairs": round(float(cos.std()), 4)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2805)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    movies, _, members, franchise = load()
    ids = sorted(movies)
    no_fr = [i for i in ids if i not in franchise]
    ln = np.array([movies[i]["y"] for i in ids])
    variants = {"all": (ids, residuals(ids, movies)), "no_franchise": (no_fr, residuals(no_fr, movies)),
                "rank": (ids, residuals(ids, movies, normal_scores(ln)))}
    summary = {"films": len(ids), "shuffles": args.shuffles, "seed": args.seed, "people_roles": PEOPLE,
               "graph": {}, "scripts": {}}
    for name, (vids, y) in variants.items():
        res = summary["graph"][name] = graph_tests(vids, y, movies, members, args.shuffles, rng)
        for part in ("distance", "dose", "company"):
            for key, r in res[part].items():
                print(f"{name:13s} {part:8s} {key:26s} pairs {r['pairs']:9d} corr {r['corr']:+.3f} "
                      f"null95 all {r['null_all_95']:+.3f} year {r['null_year_95']:+.3f} studio {r['null_studio_95']:+.3f}")
    for name in ("all", "no_franchise"):
        vids, y = variants[name]
        res = summary["scripts"][name] = script_tests(vids, y, movies, members, args.shuffles, rng)
        for key, r in res["pairs"].items():
            print(f"{name:13s} scripts  {key:30s} pairs {r['pairs']:8d} corr {r['corr']:+.3f} "
                  f"null95 all {r['null_all_95']:+.3f} year {r['null_year_95']:+.3f}")
        print(f"{name:13s} scripts  deciles", [d["corr"] for d in res["unlinked_by_cosine_decile"]])
        print(f"{name:13s} scripts  similarity", res["script_similarity"])
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
