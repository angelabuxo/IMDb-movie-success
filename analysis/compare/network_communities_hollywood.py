"""Post-hoc run 3 (declared in docs/q2-network.md before running): communities inside the full network's largest one.

network_communities_full.py found one community of 136,577 films holding 5,821 of the 6,346 usable_money films, with
the 10,000-vote network's production houses (Blumhouse, Disney animation, ...) nowhere to be seen. Modularity has a
resolution limit: in a network this size, a group of a hundred films can be too small for Louvain to split off. So
this takes that community's films and the links between them, exactly as in run 2, and repeats run 2's methods on
them alone.

  Partition  run 2's Louvain is rebuilt (same network, same seeds, checked against its saved modularity) and its
             largest community taken.
  Methods    as run 2: Louvain (igraph) over SEEDS seeds, best modularity kept, mean NMI to the other seeds; null of
             REWIRES degree-preserving rewirings with shuffled weights and the same seeds; NMI with each film's rarest
             kept genre against GENRE_PERMS shuffles; for every sub-community with at least MIN_SIZE usable_money
             films, the budget-decile shuffle on all usable films (PERMS shuffles, Holm). All reported.
  Houses     for each 10,000-vote community (network_communities.py --set usable_money) with films here: the share of
             them in their most common sub-community, so a house that survives shows up as a high share.
What this measures: whether shared people alone separate Hollywood into groups once every film counts, and whether
those groups are the houses. What it misses: studios (not in IMDb), and the same limits as run 2.
Outputs in analysis/compare/outputs/communities_hollywood/ (names and numbers only).
Run (HPC, about 1.5 hours on 25 cores):
  IMDB_DATA_DIR=/path/to/data/raw IMDB_DIR=/path/to/imdb .venv/bin/python -m analysis.compare.network_communities_hollywood --procs 25
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
from sklearn.metrics import normalized_mutual_info_score as nmi

from analysis.compare import network_communities as nc
from analysis.compare import network_communities_full as full
from analysis.compare.network_by_genre import members
from analysis.genre.q1_categories import shuffle_test
from analysis.genre.q1_genre import DATA, SEED, load, prepare

OUT = Path(__file__).resolve().parent / "outputs" / "communities_hollywood"
TOP = 3
_G = _W = None


def _one_seed(s):
    ig.set_random_number_generator(random.Random(SEED + s))
    p = _G.community_multilevel(weights=_W)
    return np.array(p.membership), _G.modularity(p.membership, weights=_W)


def seeds_parallel(G, w, n, procs):
    """full.louvain's seeds (SEED + s), run in parallel: the same partitions, one process per seed."""
    global _G, _W
    _G, _W = G, w
    with Pool(min(procs, n)) as pool:
        res = pool.map(_one_seed, range(n))
    return [r[0] for r in res], [r[1] for r in res]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--rewires", type=int, default=full.REWIRES)
    ap.add_argument("--seeds", type=int, default=full.SEEDS, help="only for a quick smoke test; the declared run uses 10")
    ap.add_argument("--procs", type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    imdb = Path(os.environ.get("IMDB_DIR", DATA / "imdb"))
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")

    # run 2's partition, rebuilt and checked
    G, w, nodes, gen, _ = full.build(imdb)
    parts, mods = seeds_parallel(G, w, full.SEEDS, a.procs)
    best = int(np.argmax(mods))
    saved = json.loads((full.OUT / "summary.json").read_text())
    assert round(mods[best], 4) == saved["modularity"], "run 2's partition doesn't reproduce"
    lab_all = parts[best]
    big_c, n_big = Counter(lab_all).most_common(1)[0]
    assert n_big == saved["largest_community_films"]
    keep_v = np.flatnonzero(lab_all == big_c)
    G.es["weight"] = list(w)
    H = G.induced_subgraph(keep_v)
    wh = np.array(H.es["weight"], float)
    hnodes = nodes[keep_v]
    full.log(t0, f"Hollywood community: {H.vcount():,} films, {H.ecount():,} links")

    # Louvain inside it, the same seeds; null with the same rewirings as run 2
    sub_parts, sub_mods = seeds_parallel(H, wh, a.seeds, a.procs)
    sb = int(np.argmax(sub_mods))
    lab = sub_parts[sb]
    seed_nmi = float(np.mean([nmi(lab, p) for i, p in enumerate(sub_parts) if i != sb])) if a.seeds > 1 else float("nan")
    full.log(t0, f"Louvain inside: modularity {sub_mods[sb]:.4f}, {len(set(lab)):,} communities, seed NMI {seed_nmi:.3f}")
    full.SEEDS, full._G, full._W = a.seeds, H, wh
    with Pool(a.procs) as pool:
        null_mod = np.array(pool.map(full._rewired, range(a.rewires), chunksize=1))
    full.log(t0, f"null: rewired modularity mean {null_mod.mean():.4f}, max {null_mod.max():.4f}")

    gi = {g: j for j, g in enumerate(keep)}
    gl = [[gi[g] for g in str(gen.get(t, "")).split(",") if g in gi] for t in hnodes]
    rar = np.bincount([j for js in gl for j in js], minlength=len(keep))
    has = np.array([bool(js) for js in gl])
    rare = np.array([min(js, key=lambda j: rar[j]) for js in gl if js])
    full._LG, full._RARE = lab[has], rare
    g_nmi = float(nmi(full._LG, rare))
    with Pool(a.procs) as pool:
        g_null = np.array(pool.map(full._genre_null, range(full.GENRE_PERMS), chunksize=10))
    full.log(t0, f"genre NMI {g_nmi:.3f} against null 95th pct {np.quantile(g_null, .95):.3f}")

    # returns: sub-communities with enough usable films, tested on all usable films as runs 1 and 2 do
    pos = {t: i for i, t in enumerate(hnodes)}
    flab = np.array([lab[pos[t]] if t in pos else -1 for t in f["imdb_id"]])
    in_h = flab >= 0
    usable_sizes = Counter(flab[in_h])
    big = [c for c, n in usable_sizes.most_common() if n >= full.MIN_SIZE]
    Wc = np.zeros((len(f), len(big)))
    for j, c in enumerate(big):
        Wc[flab == c, j] = 1
    base = M[in_h].mean(0)
    names = []
    for j, c in enumerate(big):
        Mg = M[flab == c]
        share = Mg.mean(0)
        lift = np.where(Mg.sum(0) >= 10, share / base, 0)
        k = int(np.argmax(lift))
        names.append(f"{j + 1}: {keep[k]} {share[k]:.0%} (Hollywood films {base[k]:.0%}), {usable_sizes[c]} usable films")
    tests = shuffle_test(f, f["ln_gross_over_budget"].to_numpy(), Wc, names, a.perms, SEED + 78, "Hollywood sub-community")
    full.log(t0, f"returns: {len(big)} sub-communities with at least {full.MIN_SIZE} usable films")

    # the 10,000-vote houses: where do their films land
    G10 = nc.weighted_links(f, "credits")
    n10 = sorted(G10.nodes())
    p10, m10 = nc.louvain(G10, nc.SEEDS)
    b10 = int(np.argmax(m10))
    assert round(m10[b10], 4) == json.loads((nc.out_dir("usable_money") / "summary.json").read_text())["modularity"]
    lab10 = dict(zip(n10, nc.labels(p10[b10], n10)))
    big10 = [c for c, n in Counter(lab10.values()).most_common() if n >= nc.MIN_SIZE]
    prof10 = pd.read_csv(nc.out_dir("usable_money") / "community_profiles.csv")
    sub_of = {t: int(flab[i]) for i, t in enumerate(f["imdb_id"]) if flab[i] >= 0}
    big_index = {c: j + 1 for j, c in enumerate(big)}
    houses = []
    for j, c in enumerate(big10):
        S = [t for t, l in lab10.items() if l == c]
        here = [sub_of[t] for t in S if t in sub_of]
        top, k = Counter(here).most_common(1)[0] if here else (None, 0)
        houses.append({"community_10k": j + 1, "top_studios_10k": prof10.loc[j, "top_studios"], "films_10k": len(S),
                       "films_in_hollywood": len(here), "main_sub_community": big_index.get(top, "small"),
                       "share_in_main_sub_community": round(k / len(here), 3) if here else None,
                       "excess_10k": prof10.loc[j, "excess"]})
    both = [t for t in sub_of if t in lab10]
    nmi_10k = float(nmi([sub_of[t] for t in both], [lab10[t] for t in both]))

    mem = members(set(f["imdb_id"]))
    name = pd.read_csv(DATA / "features" / "graph" / "nodes.csv", usecols=["id", "label"]).set_index("id")["label"]
    fi = f.set_index("imdb_id")
    sizes_all = Counter(lab)
    rows = []
    for j, c in enumerate(big):
        S = set(f["imdb_id"][flab == c])
        sub = fi.loc[sorted(S)]
        row = {"community": j + 1, "label": tests[j]["category"], "films_in_hollywood": int(sizes_all[c]), "usable_films": len(S),
               "year_median": int(sub.year.median()), "median_budget_musd": round(float(sub.budget.median()) / 1e6, 1),
               "median_multiple": round(float(np.exp(sub.ln_gross_over_budget.median())), 2),
               "excess": tests[j]["excess"], "p_holm": tests[j]["p_holm"]}
        for role in ("studio", "director", "producer", "actor"):
            cnt = Counter({e: len(fs & S) for e, fs in mem[role].items() if len(fs & S) >= 2})
            row[f"top_{role}s"] = "; ".join(f"{name.get(e, e)} ({k})" for e, k in cnt.most_common(TOP))
        rows.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(tests).to_csv(OUT / "community_returns.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "community_profiles.csv", index=False)
    pd.DataFrame(houses).to_csv(OUT / "houses_10k.csv", index=False)
    summary = {"films": H.vcount(), "links": H.ecount(), "usable_films_inside": int(in_h.sum()), "seeds": a.seeds,
               "louvain": "igraph community_multilevel", "communities": len(set(lab)),
               "largest_sub_community_films": int(max(sizes_all.values())),
               "modularity": round(sub_mods[sb], 4), "mean_nmi_best_vs_other_seeds": round(seed_nmi, 3),
               "rewires": a.rewires, "rewired_modularity_mean": round(float(null_mod.mean()), 4),
               "rewired_modularity_max": round(float(null_mod.max()), 4),
               "modularity_p": round(float((1 + (null_mod >= sub_mods[sb]).sum()) / (1 + a.rewires)), 4),
               "films_with_kept_genre": int(has.sum()), "nmi_with_rarest_genre": round(g_nmi, 3),
               "nmi_genre_null_95": round(float(np.quantile(g_null, .95)), 3),
               "nmi_genre_p": round(float((1 + (g_null >= g_nmi).sum()) / (1 + full.GENRE_PERMS)), 4),
               "min_size": full.MIN_SIZE, "communities_min_size": len(big), "usable_films_in_big_communities": int(Wc.sum()),
               "nmi_with_10k_communities_on_usable_films": round(nmi_10k, 3), "usable_films_in_both": len(both),
               "perms": a.perms}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    with pd.option_context("display.width", 250, "display.max_colwidth", 60):
        print(pd.DataFrame(rows).drop(columns=["label"]).to_string(index=False))
        print(pd.DataFrame(houses).to_string(index=False))
    full.log(t0, "done")


if __name__ == "__main__":
    main()
