"""Communities in the people network (Q2): are they real, do they follow genre, and do any beat their budgets?

The reproducible version of the community comparison on the film-network explorer page (explorer/graph_pages.py,
movie_network): the same four kinds of shared credit, but on the fixed film set and links of Q2's test
(scripts/network/q2_network.py), so it describes the network in the paper's Fig. 2. Course methods only (week 4
communities and modularity, week 2 nulls).

  Films     in_common_set (1,598), Q2's primary set. Films with no link are left out of the communities.
  Links     two films are linked when they share a director, one of the first three billed actors, a producer or a
            studio (network_by_genre.members, as in Q2); weight = the number of people and studios they share.
  Methods   Louvain over SEEDS seeds (the best modularity is kept; mean NMI to the other seeds = stability). Leiden
            and Infomap were dropped as second methods: they aren't course methods, and the seeds already measure
            stability.
  Null      the same network rewired with every film's degree kept (double edge swaps) and the weights shuffled,
            REWIRES times, Louvain with the same SEEDS on each; p = share of rewirings reaching the observed modularity.
  Genre     NMI of the Louvain communities with each film's rarest IMDb genre (as q1_categories does), against the
            same NMI after shuffling genres between films PERMS times.
  Returns   each community with at least MIN_SIZE films: mean ln(gross / budget) against the budget-decile shuffle of
            q1_genre (PERMS shuffles), Holm over the communities, as Q1 tests genres.
What this measures: whether shared credits split the films into groups that are tighter than chance, how much those
groups are genres, and whether any group earns differently from films of the same budget. Communities describe
the network; they don't explain why a group earns more.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.compare.network_communities [--set usable_money] [--links no_studio] [--procs 8]
      --links no_studio drops the studio links (people only), written to outputs/communities_<set>_no_studio/: a
      post-hoc check declared in src/analysis/network/Q2_NETWORK.md, next to network_communities_full.py (people from full IMDb).
      --set usable_money runs the same analysis on all 6,346 films with clean money (no script needed), written to
      outputs/communities_usable_money/. With --procs > 1 the rewirings run in parallel, each with its own seed, so
      the null differs from a --procs 1 run in its random draws (the saved common-set run used --procs 1).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi

from analysis.compare.network_by_genre import members
from analysis.genre.q1_categories import shuffle_test
from analysis.genre.q1_genre import SEED, load, prepare

OUT = Path(__file__).resolve().parent / "outputs" / "communities"
SEEDS = 10
REWIRES = 50
MIN_SIZE = 30


LINKS = {"credits": ("director", "actor", "producer", "studio"), "no_studio": ("director", "actor", "producer")}


def out_dir(flag, links="credits"):
    d = OUT if flag == "in_common_set" else OUT.parent / f"communities_{flag}"
    return d if links == "credits" else d.parent / f"{d.name}_{links}"


def _rewired_modularity(args):
    edges, w, seed = args
    R = nx.Graph()
    R.add_edges_from(edges)
    nx.double_edge_swap(R, nswap=5 * R.number_of_edges(), max_tries=10 ** 8, seed=seed)
    for (u, v), x in zip(R.edges(), np.random.default_rng(seed).permutation(w)):
        R[u][v]["weight"] = float(x)
    return max(louvain(R, SEEDS)[1])


def weighted_links(f, links="credits"):
    ids = set(f["imdb_id"])
    w = defaultdict(int)
    for role, mem in members(ids).items():
        if role not in LINKS[links]:
            continue
        for films in mem.values():
            for a, b in combinations(sorted(films & ids), 2):
                w[(a, b)] += 1
    G = nx.Graph()
    G.add_weighted_edges_from((a, b, float(x)) for (a, b), x in w.items())
    return G


def louvain(G, seeds):
    parts = [nx.community.louvain_communities(G, weight="weight", seed=s) for s in range(seeds)]
    mods = [nx.community.modularity(G, p, weight="weight") for p in parts]
    return parts, mods


def labels(part, nodes):
    lab = {n: k for k, c in enumerate(part) for n in c}
    return np.array([lab[n] for n in nodes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--set", default="in_common_set", choices=["in_common_set", "usable_money"])
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--links", default="credits", choices=list(LINKS))
    a = ap.parse_args()
    out = out_dir(a.set, a.links)
    films, genres = load()
    f, keep, M = prepare(films, genres, a.set)
    G = weighted_links(f, a.links)
    nodes = sorted(G.nodes())
    rng = np.random.default_rng(SEED)

    # Louvain, stability over seeds
    parts, mods = louvain(G, SEEDS)
    best = int(np.argmax(mods))
    lab = labels(parts[best], nodes)
    seed_nmi = float(np.mean([nmi(lab, labels(p, nodes)) for i, p in enumerate(parts) if i != best]))

    W = [d["weight"] for *_, d in G.edges(data=True)]
    methods = [{"method": "Louvain (best of seeds)", "communities": len(set(lab)), "modularity": round(mods[best], 4),
                "mean_nmi_with_other_seeds": round(seed_nmi, 3)}]

    # null: degree-preserving rewiring, weights shuffled, same Louvain seeds
    w = np.array(W)
    null_mod = []
    if a.procs == 1:
        for s in range(REWIRES):
            R = nx.Graph(G)
            nx.double_edge_swap(R, nswap=5 * R.number_of_edges(), max_tries=10 ** 7, seed=SEED + s)
            for (u, v), x in zip(R.edges(), rng.permutation(w)):
                R[u][v]["weight"] = float(x)
            null_mod.append(max(louvain(R, SEEDS)[1]))
    else:
        from multiprocessing import Pool
        edges = list(G.edges())
        with Pool(a.procs) as pool:
            null_mod = pool.map(_rewired_modularity, [(edges, w, SEED + 1000 + s) for s in range(REWIRES)])
    null_mod = np.array(null_mod)

    # genre: NMI with each film's rarest IMDb genre, against shuffled genres
    pos = {t: i for i, t in enumerate(f["imdb_id"])}
    rows_f = np.array([pos[n] for n in nodes])
    Mg = M[rows_f]
    rar = M.sum(0)
    rare = np.array([np.argmin(np.where(m > 0, rar, np.inf)) for m in Mg])
    g_nmi = float(nmi(lab, rare))
    g_null = np.array([nmi(lab, rng.permutation(rare)) for _ in range(1000)])

    # returns per community, against the budget-decile shuffle on all common-set films
    sizes = Counter(lab)
    big = [c for c, n in sizes.most_common() if n >= MIN_SIZE]
    Wc = np.zeros((len(f), len(big)))
    for j, c in enumerate(big):
        Wc[rows_f[lab == c], j] = 1
    names = []                         # label: the genre most over-represented in the community (lift), with its share
    base = Mg.mean(0)
    for j, c in enumerate(big):
        share = Mg[lab == c].mean(0)
        lift = np.where(Mg[lab == c].sum(0) >= 10, share / base, 0)
        k = int(np.argmax(lift))
        names.append(f"{j + 1}: {keep[k]} {share[k]:.0%} (all films {base[k]:.0%}), {sizes[big[j]]} films")
    tests = shuffle_test(f, f["ln_gross_over_budget"].to_numpy(), Wc, names, a.perms, SEED + 77, "people-network community")

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(methods).to_csv(out / "methods.csv", index=False)
    pd.DataFrame(tests).to_csv(out / "community_returns.csv", index=False)
    summary = {"set": a.set, "link_types": a.links, "films": len(f), "films_in_network": len(nodes), "links": G.number_of_edges(), "seeds": SEEDS,
               "communities": len(set(lab)), "communities_min_size": len(big), "min_size": MIN_SIZE,
               "films_in_big_communities": int(sum(sizes[c] for c in big)),
               "modularity": round(mods[best], 4), "mean_nmi_best_vs_other_seeds": round(seed_nmi, 3),
               "rewires": REWIRES, "rewired_modularity_mean": round(float(null_mod.mean()), 4),
               "rewired_modularity_max": round(float(null_mod.max()), 4),
               "modularity_p": round(float((1 + (null_mod >= mods[best]).sum()) / (1 + REWIRES)), 4),
               "nmi_with_rarest_genre": round(g_nmi, 3), "nmi_genre_null_95": round(float(np.quantile(g_null, .95)), 3),
               "nmi_genre_p": round(float((1 + (g_null >= g_nmi).sum()) / 1001), 4), "perms": a.perms}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(pd.DataFrame(methods).to_string(index=False))
    print(pd.DataFrame(tests).sort_values("excess").to_string(index=False))


if __name__ == "__main__":
    main()
