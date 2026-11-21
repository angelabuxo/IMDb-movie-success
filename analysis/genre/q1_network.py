"""Q1 with the course's network tools (weeks 1-4), added after the main results in q1_genre.py.

A. Genre co-occurrence network. Nodes are the kept genres; two genres are linked by the number of usable-money
   films that carry both (week 1: a projection of the film-genre bipartite network).
     Null (week 2)    curveball swaps of the film x genre matrix, which keep every film's number of genres and
                      every genre's number of films. A pair co-occurs more than chance when its count is above the
                      null's: z against the null samples, a two-sided normal p from z (Holm over all 153 pairs needs
                      p far below what a few thousand samples can resolve), and the empirical p next to it.
     Backbone (wk 4)  the links with Holm p < .05 and more films than the null expects. Second method: the
                      disparity filter (Serrano et al. 2009) at alpha = .05 on the raw counts; their overlap is
                      reported.
     Communities (4)  Louvain on the backbone (weights = observed minus expected films), over SEEDS seeds; the
                      partition with the highest modularity is kept, its agreement with the others is the mean NMI,
                      and its modularity is compared with Louvain on the backbone rewired by double edge swaps
                      (degrees kept) with its weights shuffled, 200 times.
     Returns          every genre family gets the main test of q1_genre.py. A film counts once, split across the
                      families by its share of genres in each (Action|Crime|Drama is 1/3 in each of their families),
                      and the weighted mean ln(gross / budget) is shuffled within budget deciles.
B. Genre assortativity on the film network (week 3, mixing patterns). Films are linked when they share a director,
   a writer, a producer or one of the first three billed actors (graph/credits.csv). For each genre, the
   assortativity coefficient of the 0/1 genre indicator; null: the films' genre sets shuffled between films of the
   same era bin, which keeps each film's links and each era's genre mix. Overall: the share of links whose two films
   share at least one genre, against the same null.

Outputs in analysis/genre/outputs/network/. Numbers only.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_network [--null 1000] [--perms 10000]
"""
from __future__ import annotations

import argparse
import json

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from analysis.genre.q1_genre import DATA, OUT as MAIN_OUT, SEED, holm, load, permuted_means, prepare, strata

OUT = MAIN_OUT / "network"
SEEDS = 20
ALPHA = .05
ROLES = ("director", "writer", "producer")
TOP_ACTORS = 3


# ---------- A. genre co-occurrence ----------

def curveball(rows, n_trades, rng):
    """One curveball pass (Strona et al. 2014) over a list of sets: keeps every set's size and every item's count."""
    n = len(rows)
    for _ in range(n_trades):
        a, b = rng.integers(0, n, 2)
        if a == b:
            continue
        ra, rb = rows[a], rows[b]
        only_a, only_b = list(ra - rb), list(rb - ra)
        if not only_a or not only_b:
            continue
        pool = only_a + only_b
        rng.shuffle(pool)
        k = len(only_a)
        new_a, new_b = set(pool[:k]), set(pool[k:])
        rows[a] = (ra & rb) | new_a
        rows[b] = (ra & rb) | new_b
    return rows


def cooc(rows, g):
    C = np.zeros((g, g))
    for r in rows:
        r = list(r)
        for i in r:
            for j in r:
                C[i, j] += 1
    return C


def null_cooc(M, n_null, rng):
    rows = [set(np.flatnonzero(m)) for m in M]
    rows = curveball(rows, 10 * len(rows), rng)          # burn-in
    out = np.empty((n_null, M.shape[1], M.shape[1]))
    for s in range(n_null):
        rows = curveball(rows, len(rows), rng)
        out[s] = cooc(rows, M.shape[1])
    return out


def disparity_keep(C, alpha):
    """Disparity filter: keep a link if it is significant for either endpoint."""
    W = C.copy()
    np.fill_diagonal(W, 0)
    keep = np.zeros_like(W, bool)
    for i in range(len(W)):
        k = (W[i] > 0).sum()
        if k < 2:
            continue
        p = W[i] / W[i].sum()
        a = (1 - p) ** (k - 1)
        keep[i] |= (a < alpha) & (W[i] > 0)
    return keep | keep.T


def louvain(G, seeds):
    parts = [nx.community.louvain_communities(G, weight="weight", seed=s) for s in range(seeds)]
    mods = [nx.community.modularity(G, p, weight="weight") for p in parts]
    return parts, mods


def labels(part, nodes):
    lab = {n: k for k, c in enumerate(part) for n in c}
    return [lab[n] for n in nodes]


def genre_network(f, keep, M, n_null, perms):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    rng = np.random.default_rng(SEED + 600)
    C = M.T @ M
    nl = null_cooc(M, n_null, rng)
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    iu = np.triu_indices(len(keep), 1)
    from scipy.stats import norm
    p_emp = (1 + (np.abs(nl - mu) >= np.abs(C - mu) - 1e-12).sum(0)) / (1 + n_null)
    z = (C - mu) / np.where(sd > 0, sd, np.nan)
    p = 2 * norm.sf(np.abs(np.nan_to_num(z)))
    ph = np.ones_like(p)
    ph[iu] = holm(p[iu])
    ph = np.minimum(ph, ph.T)
    back = (ph < ALPHA) & (C > mu)
    np.fill_diagonal(back, False)
    disp = disparity_keep(C, ALPHA)
    pairs = [{"genre_a": keep[i], "genre_b": keep[j], "films_both": int(C[i, j]), "expected": round(mu[i, j], 2),
              "z": round(z[i, j], 2), "p_holm": float(f"{ph[i, j]:.3g}"),
              "p_empirical": round(p_emp[i, j], 5), "backbone": bool(back[i, j]),
              "disparity_filter": bool(disp[i, j])} for i, j in zip(*iu)]

    def graph_from(Cm, mum, keepmask):
        G = nx.Graph()
        G.add_nodes_from(keep)
        for i, j in zip(*iu):
            if keepmask[i, j]:
                G.add_edge(keep[i], keep[j], weight=float(Cm[i, j] - mum[i, j]))
        return G

    G = graph_from(C, mu, back)
    parts, mods = louvain(G, SEEDS)
    best = int(np.argmax(mods))
    part = sorted(parts[best], key=lambda c: -sum(M[:, keep.index(g)].sum() for g in c))
    labs = [labels(pp, keep) for pp in parts]
    agree = float(np.mean([nmi(labs[best], l) for k, l in enumerate(labs) if k != best]))
    # null for the modularity (weeks 2 and 4): the same backbone with its links rewired (double edge swaps keep every
    # genre's degree) and its weights shuffled across links, then Louvain again
    null_mod = []
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    for s in range(200):
        Gn = nx.Graph(G)
        nx.double_edge_swap(Gn, nswap=10 * Gn.number_of_edges(), max_tries=10 ** 5, seed=SEED + s)
        for (u, v), wt in zip(Gn.edges(), rng.permutation(w)):
            Gn[u][v]["weight"] = float(wt)
        null_mod.append(max(louvain(Gn, 3)[1]))
    # returns per family: each film counts once, split across families by its share of genres in each
    y = f["ln_gross_over_budget"].to_numpy()
    n_g = np.maximum(M.sum(1), 1)
    F = np.column_stack([M[:, [keep.index(g) for g in c]].sum(1) / n_g for c in part])
    has = F.sum(1) > 0
    y, F, fs = y[has], F[has], f[has]
    obs = y @ F / F.sum(0)
    fnl = permuted_means(y, F, strata(fs, "budget"), perms, np.random.default_rng(SEED + 700))
    fmu, fsd = fnl.mean(0), fnl.std(0, ddof=1)
    fp = (1 + (np.abs(fnl - fmu) >= np.abs(obs - fmu) - 1e-12).sum(0)) / (1 + perms)
    fph = holm(fp)
    fam = [{"family": k + 1, "genres": "|".join(sorted(c)), "films_weighted": round(float(F[:, k].sum()), 1),
            "films_any": int((F[:, k] > 0).sum()), "mean": round(obs[k], 4),
            "excess": round(obs[k] - fmu[k], 4), "lo95": round(np.quantile(fnl[:, k] - fmu[k], .025), 4),
            "hi95": round(np.quantile(fnl[:, k] - fmu[k], .975), 4), "z": round((obs[k] - fmu[k]) / fsd[k], 2),
            "p_holm": round(fph[k], 5)} for k, c in enumerate(part)]
    nodes = [{"genre": g, "n_films": int(M[:, j].sum()), "family": next(k + 1 for k, c in enumerate(part) if g in c),
              "backbone_degree": int(back[j].sum())} for j, g in enumerate(keep)]
    summ = {"null_samples": n_null, "pairs": len(pairs), "backbone_links": int(back[iu].sum()),
            "disparity_links": int(disp[iu].sum()), "both": int((back & disp)[iu].sum()),
            "backbone_only": int((back & ~disp)[iu].sum()), "disparity_only": int((disp & ~back)[iu].sum()),
            "louvain_seeds": SEEDS, "modularity": round(mods[best], 4), "modularity_min_max_over_seeds": [round(min(mods), 4), round(max(mods), 4)],
            "mean_nmi_best_vs_other_seeds": round(agree, 3), "null_modularity_mean": round(float(np.mean(null_mod)), 4),
            "null_modularity_max": round(float(np.max(null_mod)), 4), "null_modularity_samples": len(null_mod),
            "modularity_p": round((1 + sum(m >= mods[best] for m in null_mod)) / (1 + len(null_mod)), 4),
            "families": len(part)}
    return pd.DataFrame(pairs), pd.DataFrame(nodes), pd.DataFrame(fam), summ, G


# ---------- B. genre assortativity on the film network ----------

def film_network(f):
    cr = pd.read_csv(DATA / "features" / "graph" / "credits.csv")
    cr = cr[cr["tconst"].isin(set(f["imdb_id"]))]
    cr = cr[cr["role"].isin(ROLES) | ((cr["role"] == "actor") & (cr["ordering"] <= TOP_ACTORS))]
    fi = {t: k for k, t in enumerate(f["imdb_id"])}
    people = {p: k for k, p in enumerate(cr["nconst"].unique())}
    B = sparse.csr_matrix((np.ones(len(cr)), ([fi[t] for t in cr["tconst"]], [people[p] for p in cr["nconst"]])),
                          shape=(len(f), len(people)))
    B.data[:] = 1
    A = (B @ B.T).tocoo()
    m = A.row < A.col
    return np.column_stack([A.row[m], A.col[m]]), len(people)


def assortativity(edges, x):
    """Newman's assortativity for a scalar node attribute on an undirected edge list."""
    a = np.concatenate([x[edges[:, 0]], x[edges[:, 1]]])
    b = np.concatenate([x[edges[:, 1]], x[edges[:, 0]]])
    sa, sb = a.std(), b.std()
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)) if sa > 0 and sb > 0 else np.nan


def film_assortativity(f, keep, M, perms):
    edges, n_people = film_network(f)
    rng = np.random.default_rng(SEED + 800)
    groups = pd.factorize(f["era_bin"])[0]
    idx_by_group = [np.flatnonzero(groups == s) for s in np.unique(groups)]

    def stats(Mx):
        r = [assortativity(edges, Mx[:, j]) for j in range(Mx.shape[1])]
        share = float(((Mx[edges[:, 0]] * Mx[edges[:, 1]]).sum(1) > 0).mean())
        return np.array(r + [share])

    obs = stats(M)
    nl = np.empty((perms, len(obs)))
    for p in range(perms):
        perm = np.arange(len(f))
        for idx in idx_by_group:
            perm[idx] = rng.permutation(idx)
        nl[p] = stats(M[perm])
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    pv = (1 + (np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12).sum(0)) / (1 + perms)
    ph = holm(pv[:-1])
    rows = [{"genre": g, "n_films": int(M[:, j].sum()), "assortativity": round(obs[j], 4), "null_mean": round(mu[j], 4),
             "null_lo95": round(np.quantile(nl[:, j], .025), 4), "null_hi95": round(np.quantile(nl[:, j], .975), 4),
             "z": round((obs[j] - mu[j]) / sd[j], 1), "p_holm": round(ph[j], 5)} for j, g in enumerate(keep)]
    deg = np.bincount(edges.ravel(), minlength=len(f))
    summ = {"films": len(f), "people": n_people, "links": len(edges), "films_with_no_link": int((deg == 0).sum()),
            "mean_degree": round(float(deg.mean()), 2), "roles": list(ROLES) + [f"actor (first {TOP_ACTORS} billed)"],
            "perms": perms, "share_links_sharing_a_genre": round(obs[-1], 4),
            "share_null_mean": round(mu[-1], 4), "share_null_hi95": round(np.quantile(nl[:, -1], .975), 4),
            "share_p": round(pv[-1], 5)}
    return pd.DataFrame(rows), summ


def figure(G, nodes, main, fam, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ex = main.set_index("genre")["excess"]
    nd = nodes.set_index("genre")
    pos = nx.spring_layout(G, weight="weight", seed=SEED, k=1.1)
    lim = float(np.abs(ex).max())
    fig, ax = plt.subplots(figsize=(7, 6))
    w = np.array([d["weight"] for *_, d in G.edges(data=True)])
    nx.draw_networkx_edges(G, pos, ax=ax, width=.4 + 3.5 * w / w.max(), edge_color="#999", alpha=.7)
    shapes = "osD^vph8"
    for k, fr in fam.iterrows():
        gs = fr["genres"].split("|")
        nx.draw_networkx_nodes(G, pos, nodelist=gs, ax=ax, node_shape=shapes[k % len(shapes)],
                               node_size=[40 + nd.loc[g, "n_films"] / 4 for g in gs],
                               node_color=[ex[g] for g in gs], cmap="RdBu", vmin=-lim, vmax=lim,
                               edgecolors="#222", linewidths=.6)
    nx.draw_networkx_labels(G, {g: (x, y - .085) for g, (x, y) in pos.items()}, ax=ax, font_size=7)
    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(-lim, lim))
    fig.colorbar(sm, ax=ax, shrink=.6, label="excess ln(gross / budget), budget-matched")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=shapes[k % len(shapes)], ls="", color="#bbb", mec="#222",
                              label=f"family {r['family']} ({r['excess']:+.2f})") for k, r in fam.iterrows()],
              fontsize=7, frameon=False, loc="lower left")
    ax.set_title("Genres linked when they share more films than a curveball null expects\n"
                 "(node size = films, colour = return over budget-matched shuffle, shape = Louvain family)", fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", type=int, default=1000)
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--assort-perms", type=int, default=1000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, nodes, fam, sa, G = genre_network(f, keep, M, a.null, a.perms)
    pairs.to_csv(OUT / "genre_pairs.csv", index=False)
    nodes.to_csv(OUT / "genre_nodes.csv", index=False)
    fam.to_csv(OUT / "genre_families.csv", index=False)
    print(json.dumps(sa), flush=True)
    print(fam.to_string(index=False), flush=True)
    ass, sb = film_assortativity(f, keep, M, a.assort_perms)
    ass.to_csv(OUT / "film_assortativity.csv", index=False)
    print(json.dumps(sb), flush=True)
    (OUT / "summary.json").write_text(json.dumps({"genre_network": sa, "film_network": sb}, indent=1))
    main_sh = pd.read_csv(MAIN_OUT / "genre_shuffle.csv")
    main_sh = main_sh[(main_sh["set"] == "usable_money") & (main_sh["outcome"] == "ln_ratio") & (main_sh["null"] == "budget")]
    figure(G, nodes, main_sh, fam, OUT / "figures" / "q1_genre_network")


if __name__ == "__main__":
    main()
