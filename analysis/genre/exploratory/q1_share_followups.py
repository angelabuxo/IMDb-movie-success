"""Q1's follow-up analyses rerun with the Tag Genome genre shares (features/genre_shares.csv, from
notebooks/tag_shares.ipynb) in place of IMDb's 0/1 tags. Added after the main results; nothing here changes them.

Films: the usable-money films with Tag Genome scores. A film counts in each genre by its share (the kept genres'
shares, renormalised to add up to 1). Every analysis is also run with the 0/1 tags on the same films, so the two
can be compared on equal footing.

  1 leak       the Tag Genome is built from user tags, ratings and reviews, so it carries audience size
               (notebooks/tag_profit.ipynb). 5-fold CV R2 of ln(IMDb votes) on budget decile + era, adding the tags
               or the shares; and the CV R2 of ln(gross / budget) with ln(votes) in the base model. Votes are partly
               the outcome, so the second is a sensitivity check, not a correction.
  2 checks     the tier heatmap, the smooth budget curve and the hit / below-budget shares of q1_checks.py, with
               shares as weights.
  3 spread     a film's effective number of genres (1 / sum of squared shares): mean return per quintile of it
               against the budget-decile shuffle. The same for IMDb's number of genres.
  4 network    soft co-occurrence (sum over films of share_a x share_b) against a null that shuffles each film's
               shares across the genres (keeps each film's mix, randomises which genres), 1000 samples; backbone
               with Holm p < .05; Louvain; agreement (NMI) with the IMDb-tag families from q1_network.py.
  5 assort.    the assortativity of each genre's share on the shared-people film network (links between films that
               both have Tag Genome scores), against shares shuffled between films of the same era.

Outputs in analysis/genre/outputs/share_followups/. Numbers only.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_share_followups [--perms 10000]
"""
from __future__ import annotations

import argparse
import json

import networkx as nx
import numpy as np
import pandas as pd

from analysis.genre import q1_checks as ck
from analysis.genre.q1_genre import DATA, OUT as MAIN_OUT, SEED, holm, load, permuted_means, prepare, strata
from analysis.genre.exploratory import EXP_OUT
from analysis.genre.exploratory.q1_network import assortativity, film_network, louvain, labels
from analysis.genre.exploratory.q1_shares import cv_r2, shares

OUT = EXP_OUT / "share_followups"


def leak(f, keep, M, G):
    v = pd.read_csv(DATA / "features" / "movies_features_10k.csv", usecols=["tconst", "votes"]).drop_duplicates("tconst")
    lv = np.log(f["imdb_id"].map(v.set_index("tconst")["votes"]).to_numpy())
    base = pd.get_dummies(pd.DataFrame({"b": f["budget_bin"].astype(str), "e": f["era_bin"]}), dtype=float).to_numpy()
    y = f["ln_gross_over_budget"].to_numpy()
    rows = []
    for target, yy, b in (("ln_votes", lv, base), ("ln_ratio", y, base), ("ln_ratio_votes_in_base", y, np.hstack([base, lv[:, None]]))):
        r0 = cv_r2(yy, b)[0]
        for name, X in (("base", b), ("+ IMDb tags", np.hstack([b, M])), ("+ Tag Genome shares", np.hstack([b, G]))):
            m, s = cv_r2(yy, X)
            rows.append({"target": target, "model": name, "cv_r2": round(m, 4), "added": round(m - r0, 4)})
    return pd.DataFrame(rows)


def spread(f, M, G, perms):
    y = f["ln_gross_over_budget"].to_numpy()
    eff = 1 / (G ** 2).sum(1)
    out = []
    for name, x in (("effective_genres_tag_genome", eff), ("imdb_n_genres", M.sum(1))):
        if name == "imdb_n_genres":
            grp = np.clip(x, 1, 3).astype(int)
            names_ = {1: "1 genre", 2: "2 genres", 3: "3 genres"}
        else:
            grp = pd.qcut(x, 5, labels=False)
            edges = np.quantile(x, np.linspace(0, 1, 6))
            names_ = {k: f"q{k + 1} ({edges[k]:.2f}-{edges[k + 1]:.2f})" for k in range(5)}
        ks = sorted(set(grp))
        W = np.column_stack([(grp == k).astype(float) for k in ks])
        obs = y @ W / W.sum(0)
        nl = permuted_means(y, W, strata(f, "budget"), perms, np.random.default_rng(SEED + 1100 + len(out)))
        mu, sd, p = ck.stats(obs, nl, perms)
        ph = holm(p)
        out += [{"measure": name, "group": names_[k], "films": int(W[:, i].sum()), "mean": round(obs[i], 4),
                 "excess": round(obs[i] - mu[i], 4), "z": round((obs[i] - mu[i]) / sd[i], 2), "p_holm": round(ph[i], 5)}
                for i, k in enumerate(ks)]
    from scipy.stats import spearmanr
    net = lambda v: v - pd.Series(v).groupby(f["budget_bin"].to_numpy()).transform("mean").to_numpy()
    rho = spearmanr(eff, net(y))
    votes = pd.read_csv(DATA / "features" / "movies_features_10k.csv", usecols=["tconst", "votes"]).drop_duplicates("tconst")
    lv = np.log(f["imdb_id"].map(votes.set_index("tconst")["votes"]).to_numpy())
    rv = spearmanr(eff, net(lv))
    rn = spearmanr(M.sum(1), net(lv))
    return pd.DataFrame(out), {"spearman_effective_genres_vs_budget_netted_return": round(float(rho[0]), 3),
                               "p": float(f"{rho[1]:.3g}"),
                               "spearman_effective_genres_vs_budget_netted_ln_votes": round(float(rv[0]), 3),
                               "spearman_imdb_n_genres_vs_budget_netted_ln_votes": round(float(rn[0]), 3)}


def share_network(keep, G, n_null):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from scipy.stats import norm
    rng = np.random.default_rng(SEED + 1200)
    C = G.T @ G
    nl = np.empty((n_null, len(keep), len(keep)))
    for s in range(n_null):
        Gp = np.take_along_axis(G, np.argsort(rng.random(G.shape), axis=1), axis=1)
        nl[s] = Gp.T @ Gp
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    z = (C - mu) / sd
    iu = np.triu_indices(len(keep), 1)
    ph = np.ones_like(z)
    ph[iu] = holm(2 * norm.sf(np.abs(z[iu])))
    ph = np.minimum(ph, ph.T)
    back = (ph < .05) & (C > mu)
    np.fill_diagonal(back, False)
    Gr = nx.Graph()
    Gr.add_nodes_from(keep)
    for i, j in zip(*iu):
        if back[i, j]:
            Gr.add_edge(keep[i], keep[j], weight=float(C[i, j] - mu[i, j]))
    parts, mods = louvain(Gr, 20)
    best = int(np.argmax(mods))
    lab = labels(parts[best], keep)
    imdb_nodes = pd.read_csv(EXP_OUT / "network" / "genre_nodes.csv").set_index("genre").reindex(keep)
    imdb_pairs = pd.read_csv(EXP_OUT / "network" / "genre_pairs.csv")
    ib = {frozenset((a, b)) for a, b, k in zip(imdb_pairs["genre_a"], imdb_pairs["genre_b"], imdb_pairs["backbone"]) if k}
    sb = {frozenset((keep[i], keep[j])) for i, j in zip(*iu) if back[i, j]}
    pairs = pd.DataFrame([{"genre_a": keep[i], "genre_b": keep[j], "soft_cooc": round(C[i, j], 2), "expected": round(mu[i, j], 2),
                           "z": round(z[i, j], 2), "p_holm": float(f"{ph[i, j]:.3g}"), "backbone": bool(back[i, j]),
                           "in_imdb_backbone": frozenset((keep[i], keep[j])) in ib} for i, j in zip(*iu)])
    fam = pd.DataFrame([{"family": k + 1, "genres": "|".join(sorted(c))} for k, c in enumerate(sorted(parts[best], key=len, reverse=True))])
    summ = {"null_samples": n_null, "backbone_links": len(sb), "imdb_backbone_links": len(ib), "shared_links": len(sb & ib),
            "families": len(parts[best]), "modularity": round(mods[best], 4),
            "nmi_with_imdb_families": round(float(nmi(lab, imdb_nodes["family"].to_numpy())), 3)}
    return pairs, fam, summ


def share_assort(f, keep, M, G, perms):
    edges, _ = film_network(f)
    rng = np.random.default_rng(SEED + 1300)
    groups = pd.factorize(f["era_bin"])[0]
    idx = [np.flatnonzero(groups == s) for s in np.unique(groups)]
    out = []
    for rep, X in (("imdb_tags", M), ("tag_genome_shares", G)):
        obs = np.array([assortativity(edges, X[:, j]) for j in range(len(keep))])
        nl = np.empty((perms, len(keep)))
        for p in range(perms):
            perm = np.arange(len(f))
            for ix in idx:
                perm[ix] = rng.permutation(ix)
            nl[p] = [assortativity(edges, X[perm, j]) for j in range(len(keep))]
        mu, sd, pv = ck.stats(obs, nl, perms)
        ph = holm(pv)
        out += [{"representation": rep, "genre": g, "assortativity": round(obs[j], 4), "null_hi95": round(np.quantile(nl[:, j], .975), 4),
                 "z": round((obs[j] - mu[j]) / sd[j], 1), "p_holm": round(ph[j], 5)} for j, g in enumerate(keep)]
    return pd.DataFrame(out), {"films": len(f), "links": len(edges)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--null", type=int, default=1000)
    ap.add_argument("--assort-perms", type=int, default=1000)
    a = ap.parse_args()
    films, genres = load()
    f0, keep, M0 = prepare(films, genres, "usable_money")
    G0, has = shares("genre_shares.csv", f0, keep)
    f, M = f0[has].reset_index(drop=True), M0[has]
    G = G0[has] / G0[has].sum(1, keepdims=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"films": len(f), "genres": len(keep), "perms": a.perms}
    lk = leak(f, keep, M, G)
    lk.to_csv(OUT / "leak.csv", index=False)
    print(lk.to_string(index=False), flush=True)
    for rep, W in (("imdb_tags", M), ("tag_genome_shares", G)):
        ck.tiers(f, keep, W, a.perms).assign(representation=rep).to_csv(OUT / f"tiers_{rep}.csv", index=False)
        pd.DataFrame(ck.smooth(f, keep, W, a.perms)).assign(representation=rep).to_csv(OUT / f"smooth_{rep}.csv", index=False)
        pd.DataFrame(ck.shares(f, keep, W, a.perms)).assign(representation=rep).to_csv(OUT / f"hitshare_{rep}.csv", index=False)
    sp, rho = spread(f, M, G, a.perms)
    sp.to_csv(OUT / "spread.csv", index=False)
    summ["spread"] = rho
    print(sp.to_string(index=False), rho, flush=True)
    pairs, fam, sn = share_network(keep, G, a.null)
    pairs.to_csv(OUT / "share_network_pairs.csv", index=False)
    fam.to_csv(OUT / "share_network_families.csv", index=False)
    summ["share_network"] = sn
    print(sn, fam.to_string(index=False), flush=True)
    asr, sa = share_assort(f, keep, M, G, a.assort_perms)
    asr.to_csv(OUT / "assortativity.csv", index=False)
    summ["film_network"] = sa
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    # heatmap rows in the order of the main test
    main_sh = pd.read_csv(MAIN_OUT / "genre_shuffle.csv")
    main_sh = main_sh[(main_sh["set"] == "usable_money") & (main_sh["outcome"] == "ln_ratio") & (main_sh["null"] == "budget")]
    ck.heatmap(pd.read_csv(OUT / "tiers_tag_genome_shares.csv"), main_sh, OUT / "figures" / "q1_tiers_tag_genome_shares")
    print("done", flush=True)


if __name__ == "__main__":
    main()
