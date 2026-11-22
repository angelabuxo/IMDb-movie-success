"""How stable are the 13 keyword categories of q1_categories.py, one category at a time? Added after the results.

q1_categories.py checks the network as a whole (modularity against a rewired backbone, mean NMI over Louvain seeds). This
script asks whether each category comes back, and whether its return result comes back with it.

  reference   the backbone and best Louvain partition rebuilt exactly as q1_categories.py does (same seeds), and checked
              against its saved keyword_categories.csv
  matching    a reference category is matched to the community of another partition with the highest Jaccard overlap,
              counting only keywords present in both networks
  checks
    seeds       the other 19 Louvain seeds on the reference backbone
    null draws  DRAWS backbones on all films, each from a different curveball null draw (the null is random, so the pairs
                near the Holm cut-off depend on the draw)
    films       BOOT random subsamples of 80 % of the usable-money films (without replacement), each with its own backbone
                (curveball null, MIN_KW scaled to 0.8 x 30 = 24), Louvain partition, budget deciles and budget-decile shuffle
    settings    on all films, one fixed choice changed at a time: MIN_KW 20 and 40, Louvain resolution 0.8 and 1.2, backbone
                Holm alpha 0.01
    modularity  the rewired null of q1_categories.py with 20 Louvain seeds per rewired network instead of 2, so the observed
                and null modularity are both the best of 20
  per run, the matched community's mean return against the budget-decile shuffle (PERMS shuffles, Holm over that run's
  categories)

What this measures: whether the partition and its return results depend on the Louvain seed, on which films happen to be
in the sample, or on the fixed thresholds. It doesn't test whether the categories mean anything beyond keyword co-use.
Outputs in analysis/genre/outputs/category_stability/, numbers and keyword labels only.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_category_stability [--boot 50] [--procs 8]
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm

from analysis.genre import q1_categories as qc
from analysis.genre.exploratory.q1_network import curveball
from analysis.genre.q1_genre import OUT as MAIN_OUT, SEED, holm, load, prepare

OUT = MAIN_OUT / "category_stability"
SEEDS = 20
NULL = 200
PERMS = 2000
FRAC = 0.8
MIN_KW = 30
MIN_CAT = 3


def louvain(G, seeds, resolution=1.0):
    parts = [nx.community.louvain_communities(G, weight="weight", seed=s, resolution=resolution) for s in range(seeds)]
    mods = [nx.community.modularity(G, p, weight="weight", resolution=resolution) for p in parts]
    return parts, mods


def backbone(f, min_kw, n_null, alpha, rng_seed):
    """The keyword backbone of q1_categories.part_keywords, with its thresholds as arguments."""
    qc.MIN_KW = min_kw
    vocab, rows, _ = qc.keywords(f)
    k = len(vocab)
    X = qc.to_sparse(rows, k)
    C = (X.T @ X).toarray()
    rng = np.random.default_rng(rng_seed)
    work = curveball([set(s) for s in rows], 10 * len(rows), rng)
    s1, s2 = np.zeros((k, k)), np.zeros((k, k))
    for _ in range(n_null):
        work = curveball(work, len(work), rng)
        Xn = qc.to_sparse(work, k)
        Cn = (Xn.T @ Xn).toarray()
        s1 += Cn
        s2 += Cn ** 2
    mu = s1 / n_null
    sd = np.sqrt(np.maximum(s2 / n_null - mu ** 2, 1e-12) * n_null / (n_null - 1))
    z = (C - mu) / sd
    iu = np.triu_indices(k, 1)
    ph = np.ones_like(z)
    ph[iu] = holm(2 * norm.sf(np.abs(z[iu])))
    back = (ph < alpha) & (C > mu)
    np.fill_diagonal(back, False)
    G = nx.Graph()
    G.add_nodes_from(vocab)
    for i, j in zip(*iu):
        if back[i, j]:
            G.add_edge(vocab[i], vocab[j], weight=float(C[i, j] - mu[i, j]))
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    return G, vocab, X


def category_returns(f, vocab, X, part, perms, seed):
    """Each community's weighted mean return against the budget-decile shuffle, as in q1_categories.part_keywords."""
    kix = {n: i for i, n in enumerate(vocab)}
    F = np.zeros((len(f), len(part)))
    for c_i, c in enumerate(part):
        F[:, c_i] = np.asarray(X[:, [kix[n] for n in c]].sum(1)).ravel()
    tot = F.sum(1, keepdims=True)
    has = tot[:, 0] >= 2
    W = np.where(tot > 0, F / np.maximum(tot, 1), 0)[has]
    rows = qc.shuffle_test(f[has].reset_index(drop=True), f["ln_gross_over_budget"].to_numpy()[has], W,
                           [str(i) for i in range(len(part))], perms, seed, "")
    return [(r["excess"], r["p_holm"]) for r in rows]


def match(ref, ref_nodes, part, nodes):
    """For each reference category, (best Jaccard, index of that community, share of its keywords in this network)."""
    out = []
    for c in ref:
        c2 = c & nodes
        best, arg = 0.0, None
        for j, p in enumerate(part):
            p2 = p & ref_nodes
            u = len(c2 | p2)
            jac = len(c2 & p2) / u if u else 0.0
            if jac > best:
                best, arg = jac, j
        out.append((best, arg, len(c2) / len(c)))
    return out


def one_run(args):
    """One partition: a film subsample or a changed setting. Returns per-reference-category rows."""
    label, films, genres, ref, ref_nodes, min_kw, alpha, resolution, frac, seed, n_null, perms, null_seed = args
    fu = films[films["usable_money"] & films["ln_gross_over_budget"].notna()]
    if frac < 1:
        fu = fu.sample(frac=frac, random_state=seed)
    f, _, _ = prepare(fu, genres, "usable_money")
    G, vocab, X = backbone(f, min_kw, n_null, alpha, null_seed)
    parts, mods = louvain(G, SEEDS, resolution)
    part = sorted([c for c in parts[int(np.argmax(mods))] if len(c) >= MIN_CAT], key=len, reverse=True)
    ret = category_returns(f, vocab, X, part, perms, seed + 1)
    rows = []
    for k, (jac, j, cov) in enumerate(match(ref, ref_nodes, part, set(G))):
        exc, p = ret[j] if j is not None else (np.nan, np.nan)
        rows.append({"check": label, "seed": seed, "category": k + 1, "jaccard": round(jac, 3), "coverage": round(cov, 3),
                     "matched_size": len(part[j]) if j is not None else 0, "excess": exc, "p_holm": p,
                     "categories_in_run": len(part), "films": len(f)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=50)
    ap.add_argument("--null", type=int, default=NULL)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--perms", type=int, default=PERMS)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("LSB_DJOB_NUMPROC", os.cpu_count() or 1)))
    a = ap.parse_args()
    films, genres = load()
    f, _, _ = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)

    # reference: rebuilt as q1_categories.py does, and checked against its saved categories
    G, vocab, X = backbone(f, MIN_KW, a.null, .05, SEED + 4100)
    parts, mods = louvain(G, SEEDS)
    best = int(np.argmax(mods))
    ref = sorted([c for c in parts[best] if len(c) >= MIN_CAT], key=len, reverse=True)
    saved = pd.read_csv(MAIN_OUT / "categories" / "keyword_categories.csv")
    strength = dict(G.degree(weight="weight"))
    tops = [", ".join(sorted(c, key=lambda n: -strength[n])[:8]) for c in ref]
    reproduced = len(ref) == len(saved) and tops == saved["top_keywords"].tolist()
    print("reference reproduced:", reproduced, len(ref), "categories", flush=True)
    ref_nodes = set(G)
    ref_ret = category_returns(f, vocab, X, ref, a.perms, SEED + 4200)

    rows = []
    # 1. the other Louvain seeds on the same backbone
    for s, p in enumerate(parts):
        if s == best:
            continue
        p = [c for c in p if len(c) >= MIN_CAT]
        for k, (jac, j, cov) in enumerate(match(ref, ref_nodes, p, ref_nodes)):
            rows.append({"check": "Louvain seeds", "seed": s, "category": k + 1, "jaccard": round(jac, 3), "coverage": cov,
                         "matched_size": len(p[j]) if j is not None else 0, "excess": np.nan, "p_holm": np.nan,
                         "categories_in_run": len(p), "films": len(f)})

    # 2. film subsamples and 3. changed settings, in parallel
    jobs = [("80 % of films", films, genres, ref, ref_nodes, round(MIN_KW * FRAC), .05, 1.0, FRAC, SEED + 9000 + b, a.null, a.perms, SEED + 9000 + b)
            for b in range(a.boot)]
    jobs += [("null draws", films, genres, ref, ref_nodes, MIN_KW, .05, 1.0, 1.0, SEED + 9500 + d, a.null, a.perms, SEED + 9500 + d)
             for d in range(a.draws)]
    jobs += [(lab, films, genres, ref, ref_nodes, mk, al, res, 1.0, SEED + 9900 + i, a.null, a.perms, SEED + 4100) for i, (lab, mk, al, res) in enumerate([
        ("MIN_KW 20", 20, .05, 1.0), ("MIN_KW 40", 40, .05, 1.0), ("resolution 0.8", MIN_KW, .05, .8),
        ("resolution 1.2", MIN_KW, .05, 1.2), ("backbone alpha 0.01", MIN_KW, .01, 1.0)])]
    with Pool(a.procs) as pool:
        for r in pool.imap_unordered(one_run, jobs):
            rows.extend(r)
            print(r[0]["check"], r[0]["seed"], "done", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "runs.csv", index=False)

    # 4. modularity null with as many Louvain seeds as the observed partition
    rng = np.random.default_rng(SEED + 4300)
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    null_mod = []
    for s in range(50):
        Gn = nx.Graph(G)
        nx.double_edge_swap(Gn, nswap=5 * Gn.number_of_edges(), max_tries=10 ** 6, seed=SEED + s)
        for (u, v), wt in zip(Gn.edges(), rng.permutation(w)):
            Gn[u][v]["weight"] = float(wt)
        null_mod.append(max(louvain(Gn, SEEDS)[1]))

    # per category, per check
    summ_rows = []
    for k, c in enumerate(ref):
        top = tops[k].split(", ")[0]
        base = {"category": f"{k + 1}: {top}", "keywords": len(c), "excess_all_films": ref_ret[k][0],
                "p_holm_all_films": ref_ret[k][1]}
        for chk, d in runs[runs["category"] == k + 1].groupby("check", sort=False):
            same = (np.sign(d["excess"]) == np.sign(ref_ret[k][0]))
            summ_rows.append({**base, "check": chk, "runs": len(d), "jaccard_mean": round(d["jaccard"].mean(), 3),
                              "jaccard_min": round(d["jaccard"].min(), 3), "share_jaccard_ge_0.5": round((d["jaccard"] >= .5).mean(), 3),
                              "excess_median": round(d["excess"].median(), 4) if d["excess"].notna().any() else np.nan,
                              "share_same_sign": round(same.mean(), 3) if d["excess"].notna().any() else np.nan,
                              "share_same_sign_holm_05": round((same & (d["p_holm"] < .05)).mean(), 3) if d["excess"].notna().any() else np.nan})
    per = pd.DataFrame(summ_rows)
    per.to_csv(OUT / "per_category.csv", index=False)
    summary = {"reference_reproduced": bool(reproduced), "reference_categories": len(ref), "reference_modularity": round(mods[best], 4),
               "boot": a.boot, "null_draws": a.draws, "frac": FRAC, "null": a.null, "perms": a.perms, "louvain_seeds": SEEDS,
               "rewired_modularity_best_of_20_mean": round(float(np.mean(null_mod)), 4),
               "rewired_modularity_best_of_20_max": round(float(np.max(null_mod)), 4),
               "modularity_p_best_of_20": round(float((1 + sum(m >= mods[best] for m in null_mod)) / (1 + len(null_mod))), 4),
               "categories_per_subsample_median": float(runs[runs["check"] == "80 % of films"].groupby("seed")["categories_in_run"].first().median())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(per.to_string(index=False))


if __name__ == "__main__":
    main()
