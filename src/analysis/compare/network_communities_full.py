"""Post-hoc robustness run (declared in src/analysis/network/Q2_NETWORK.md before running): the people network's communities built
from all of IMDb, not only films with 10,000+ votes.

network_communities.py links the 6,346 usable_money films through shared directors, top-three actors, producers
and studios, all taken from the movie graph, so every link runs through a noticed film. Here the network holds
every non-adult film in IMDb's title.principals (titleType movie), linked when two films share a director, one of
their first three billed actors or actresses, or a producer, weighted by the number of shared people. IMDb doesn't
record studios, so there are no studio links: this also asks whether people alone form the "production houses".
Its twin, network_communities.py --links no_studio, drops the studio links from the 6,346-film network, so the
effect of dropping studios can be told apart from the effect of adding unnoticed films.

  Methods   Louvain (igraph's community_multilevel, the same algorithm as networkx's, which takes about 85 minutes
            per run on 23 million links) over SEEDS seeds; the best modularity is kept, mean NMI to the other seeds
            = stability.
  Null      the network rewired with every film's degree kept (igraph rewire, 5 x links swaps) and the weights
            shuffled, REWIRES times, Louvain with SEEDS seeds on each; p = share of rewirings reaching the observed
            modularity.
  Genre     NMI of the communities with each film's rarest IMDb genre (among the 18 genres of the usable_money
            set; films with none of them are left out), against GENRE_PERMS shuffles of the genres.
  Returns   only on the usable_money films (the only films with clean money): every community with at least
            MIN_SIZE of them, mean ln(gross / budget) against the budget-decile shuffle (PERMS shuffles), Holm over
            the communities, as network_communities.py does. Every such community is reported.
  Profiles  for each tested community: its films in the whole network, its usable films' years, median budget and
            multiple, the directors, producers and top-three actors credited on most of its usable films, their
            most frequent studios (described, not used for links), and the 10,000-vote community (credits links,
            network_communities.py --set usable_money) most of its usable films fall in.
What this measures: whether shared people split the films into groups tighter than chance once unnoticed films
count, and whether those groups still earn differently from films of the same budget. What it misses: studios,
and credits beyond IMDb's principal ones (a median of 7 rows on unrated films, 21 on 10,000-vote films).
Outputs in analysis/compare/outputs/communities_full_imdb/ (names and numbers only).
Run (HPC: about 2 hours on 16 cores, 6 GB per process):
  IMDB_DATA_DIR=/path/to/data/raw IMDB_DIR=/path/to/imdb .venv/bin/python -m analysis.compare.network_communities_full --procs 16
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import normalized_mutual_info_score as nmi

from analysis.compare import network_communities as nc
from analysis.compare.network_by_genre import members
from analysis.genre.q1_categories import shuffle_test
from analysis.genre.q1_genre import DATA, SEED, load, prepare

OUT = Path(__file__).resolve().parent / "outputs" / "communities_full_imdb"
SEEDS = nc.SEEDS
REWIRES = nc.REWIRES
MIN_SIZE = nc.MIN_SIZE
GENRE_PERMS = 1000
TOP_ACTORS = 3
TOP = 3
_G = None          # the network, shared with the worker processes by fork
_W = None
_LG = _RARE = None


def log(t0, msg):
    print(f"[{time.time() - t0:7.0f} s] {msg}", flush=True)


def build(imdb):
    """Film x person credits from title.principals, then film-film links weighted by shared people."""
    tb = pd.read_csv(imdb / "title.basics.tsv.gz", sep="\t", usecols=["tconst", "titleType", "isAdult", "genres"],
                     dtype=str, na_values="\\N", quoting=3)
    tb = tb[(tb["titleType"] == "movie") & (tb["isAdult"] != "1")]
    gen = dict(zip(tb["tconst"], tb["genres"]))
    ch = []
    for c in pd.read_csv(imdb / "title.principals.tsv.gz", sep="\t", usecols=["tconst", "ordering", "nconst", "category"],
                         dtype=str, chunksize=5_000_000, quoting=3):
        ch.append(c[c["category"].isin(["director", "producer", "actor", "actress"]) & c["tconst"].isin(gen.keys())])
    pr = pd.concat(ch, ignore_index=True)
    pr["ordering"] = pd.to_numeric(pr["ordering"], errors="coerce")
    act = pr[pr["category"].isin(["actor", "actress"])].sort_values(["tconst", "ordering"]).groupby("tconst").head(TOP_ACTORS)
    cr = pd.concat([pr[pr["category"].isin(["director", "producer"])], act]).drop_duplicates(["tconst", "nconst"])
    films = cr["tconst"].unique()
    fi = {t: i for i, t in enumerate(films)}
    pi = {p: i for i, p in enumerate(cr["nconst"].unique())}
    A = sparse.csr_matrix((np.ones(len(cr), np.float32), (cr["tconst"].map(fi), cr["nconst"].map(pi))),
                          shape=(len(films), len(pi)))
    C = sparse.triu(A @ A.T, k=1).tocoo()
    linked = np.unique(np.concatenate([C.row, C.col]))          # films with at least one link
    remap = -np.ones(len(films), np.int64)
    remap[linked] = np.arange(len(linked))
    G = ig.Graph(n=len(linked), edges=np.column_stack([remap[C.row], remap[C.col]]).tolist())
    w = C.data.astype(float)
    info = {"imdb_movies_non_adult": len(gen), "credits": len(cr), "people": len(pi), "films_with_credits": len(films)}
    return G, w, films[linked], gen, info


def louvain(G, w, seeds, offset=0):
    parts, mods = [], []
    for s in range(seeds):
        ig.set_random_number_generator(random.Random(SEED + offset + s))
        p = G.community_multilevel(weights=w)
        parts.append(np.array(p.membership))
        mods.append(G.modularity(p.membership, weights=w))
    return parts, mods


def _rewired(s):
    R = _G.copy()
    ig.set_random_number_generator(random.Random(SEED + 5000 + s))
    R.rewire(n=5 * R.ecount())
    w = np.random.default_rng(SEED + 5000 + s).permutation(_W)
    return max(louvain(R, w, SEEDS, offset=6000 + 100 * s)[1])


def _genre_null(s):
    return nmi(_LG, np.random.default_rng(SEED + 9000 + s).permutation(_RARE))


def main():
    global _G, _W, _LG, _RARE, SEEDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--rewires", type=int, default=REWIRES)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=SEEDS, help="only for a quick smoke test; the declared run uses 10")
    a = ap.parse_args()
    SEEDS = a.seeds
    t0 = time.time()
    imdb = Path(os.environ.get("IMDB_DIR", DATA / "imdb"))
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    G, w, nodes, gen, info = build(imdb)
    log(t0, f"network: {G.vcount():,} linked films, {G.ecount():,} links")

    parts, mods = louvain(G, w, SEEDS)
    best = int(np.argmax(mods))
    lab = parts[best]
    seed_nmi = float(np.mean([nmi(lab, p) for i, p in enumerate(parts) if i != best]))
    log(t0, f"Louvain: modularity {mods[best]:.4f}, {len(set(lab)):,} communities, seed NMI {seed_nmi:.3f}")

    _G, _W = G, w
    with Pool(a.procs) as pool:
        null_mod = np.array(pool.map(_rewired, range(a.rewires), chunksize=1))
    log(t0, f"null: rewired modularity mean {null_mod.mean():.4f}, max {null_mod.max():.4f}")
    # genre: each film's rarest kept genre, rarity counted over the network's films
    gi = {g: j for j, g in enumerate(keep)}
    gl = [[gi[g] for g in str(gen.get(t, "")).split(",") if g in gi] for t in nodes]
    rar = np.bincount([j for js in gl for j in js], minlength=len(keep))
    has = np.array([bool(js) for js in gl])
    rare = np.array([min(js, key=lambda j: rar[j]) for js in gl if js])
    _LG, _RARE = lab[has], rare
    g_nmi = float(nmi(_LG, rare))
    with Pool(a.procs) as pool:
        g_null = np.array(pool.map(_genre_null, range(GENRE_PERMS), chunksize=10))
    log(t0, f"genre NMI {g_nmi:.3f} against null 95th pct {np.quantile(g_null, .95):.3f}")

    # returns on the usable films only
    pos = {t: i for i, t in enumerate(nodes)}
    in_net = np.array([t in pos for t in f["imdb_id"]])
    flab = np.array([lab[pos[t]] if t in pos else -1 for t in f["imdb_id"]])
    usable_sizes = Counter(flab[in_net])
    big = [c for c, n in usable_sizes.most_common() if n >= MIN_SIZE]
    Wc = np.zeros((len(f), len(big)))
    for j, c in enumerate(big):
        Wc[flab == c, j] = 1
    base = M[in_net].mean(0)
    names = []
    for j, c in enumerate(big):
        Mg = M[flab == c]
        share = Mg.mean(0)
        lift = np.where(Mg.sum(0) >= 10, share / base, 0)
        k = int(np.argmax(lift))
        names.append(f"{j + 1}: {keep[k]} {share[k]:.0%} (all films {base[k]:.0%}), {usable_sizes[c]} usable films")
    tests = shuffle_test(f, f["ln_gross_over_budget"].to_numpy(), Wc, names, a.perms, SEED + 77, "full-IMDb people community")
    log(t0, f"returns: {len(big)} communities with at least {MIN_SIZE} usable films")

    # profiles, and the 10,000-vote community (credits links) most of each community's usable films fall in
    G10 = nc.weighted_links(f, "credits")
    n10 = sorted(G10.nodes())
    p10, m10 = nc.louvain(G10, nc.SEEDS)
    b10 = int(np.argmax(m10))
    saved = json.loads((nc.out_dir("usable_money") / "summary.json").read_text())
    assert round(m10[b10], 4) == saved["modularity"], "10k partition differs from network_communities.py's saved run"
    lab10 = dict(zip(n10, nc.labels(p10[b10], n10)))
    big10 = [c for c, n in Counter(lab10.values()).most_common() if n >= MIN_SIZE]
    prof10 = pd.read_csv(nc.out_dir("usable_money") / "community_profiles.csv")
    both = [t for t in f["imdb_id"] if t in pos and t in lab10]
    nmi_10k = float(nmi([lab[pos[t]] for t in both], [lab10[t] for t in both]))
    sizes_all = Counter(lab)
    mem = members(set(f["imdb_id"]))
    name = pd.read_csv(DATA / "features" / "graph" / "nodes.csv", usecols=["id", "label"]).set_index("id")["label"]
    fi = f.set_index("imdb_id")
    rows = []
    for j, c in enumerate(big):
        S = set(f["imdb_id"][flab == c])
        sub = fi.loc[sorted(S)]
        c10 = Counter(lab10[t] for t in S if t in lab10)
        top10, k10 = c10.most_common(1)[0] if c10 else (None, 0)
        row = {"community": j + 1, "label": tests[j]["category"], "films_in_network": int(sizes_all[c]), "usable_films": len(S),
               "year_q25": int(sub.year.quantile(.25)), "year_median": int(sub.year.median()), "year_q75": int(sub.year.quantile(.75)),
               "median_budget_musd": round(float(sub.budget.median()) / 1e6, 1),
               "median_multiple": round(float(np.exp(sub.ln_gross_over_budget.median())), 2),
               "excess": tests[j]["excess"], "p_holm": tests[j]["p_holm"],
               "main_10k_community": (big10.index(top10) + 1) if top10 in big10 else "",
               "main_10k_share": round(k10 / len(S), 3),
               "main_10k_top_studios": prof10.loc[big10.index(top10), "top_studios"] if top10 in big10 else ""}
        for role in ("studio", "director", "producer", "actor"):
            cnt = Counter({e: len(fs & S) for e, fs in mem[role].items() if len(fs & S) >= 2})
            row[f"top_{role}s"] = "; ".join(f"{name.get(e, e)} ({k})" for e, k in cnt.most_common(TOP))
        rows.append(row)
    prof = pd.DataFrame(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tests).to_csv(OUT / "community_returns.csv", index=False)
    prof.to_csv(OUT / "community_profiles.csv", index=False)
    summary = {**info, "films_in_network": G.vcount(), "links": G.ecount(), "seeds": SEEDS, "louvain": "igraph community_multilevel",
               "communities": len(set(lab)), "largest_community_films": int(max(sizes_all.values())),
               "modularity": round(mods[best], 4), "mean_nmi_best_vs_other_seeds": round(seed_nmi, 3),
               "rewires": a.rewires, "rewired_modularity_mean": round(float(null_mod.mean()), 4),
               "rewired_modularity_max": round(float(null_mod.max()), 4),
               "modularity_p": round(float((1 + (null_mod >= mods[best]).sum()) / (1 + a.rewires)), 4),
               "films_with_kept_genre": int(has.sum()), "nmi_with_rarest_genre": round(g_nmi, 3),
               "nmi_genre_null_95": round(float(np.quantile(g_null, .95)), 3),
               "nmi_genre_p": round(float((1 + (g_null >= g_nmi).sum()) / (1 + GENRE_PERMS)), 4),
               "usable_films": len(f), "usable_films_in_network": int(in_net.sum()), "min_size": MIN_SIZE,
               "communities_min_size": len(big), "usable_films_in_big_communities": int(Wc.sum()),
               "nmi_with_10k_communities_on_usable_films": round(nmi_10k, 3), "usable_films_in_both_networks": len(both),
               "perms": a.perms}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(prof.drop(columns=["label"]).to_string(index=False))
    log(t0, "done")


if __name__ == "__main__":
    main()
