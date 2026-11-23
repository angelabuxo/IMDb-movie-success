"""Q1 with weighted and text-derived genres (course weeks 6-7), added after the main results in q1_genre.py.

IMDb gives a film up to three genres, unweighted and in alphabetical order. Two other ways to say what genre a
film is, both already in data/raw/features:
  Tag Genome shares   genre_shares.csv: the MovieLens Tag Genome's genre tag scores, turned into shares that add up
                      to 1 per film (archive/scripts/reviews/tag_shares.py). Built from what users tagged after release.
  Plot shares         plot_genre_shares.csv: the same shares predicted from the Wikipedia plot's embedding
                      (archive/scripts/reviews/plot_tags.py). Available for almost every film, written after release too.
  Plot clusters       k-means on the plot embeddings (features/plot_vectors.npy, all-MiniLM-L6-v2), with as many
                      clusters as kept IMDb genres: a genre made from text alone, with no labels (week 7: one vector
                      per text, then different ways to split the same corpus).

Tests, on the usable-money films that have each representation:
  1 Shares as weights  the main test with each film counted in each genre by its share instead of 0/1 (budget-decile
                       shuffle). Compared with the 0/1 test on the same films.
  2 Dominant genre     every film gets the one genre with its largest share, so the genres no longer overlap and the
                       tests are separate; same shuffle.
  3 Agreement          NMI (week 4) between the dominant Tag Genome genre, the dominant plot genre and the plot
                       clusters; and how often each dominant genre is one of the film's IMDb genres, against the same
                       dominant genres handed to random films. (The clusters are described by the IMDb genre with the
                       largest Tag Genome share among each film's IMDb genres, for naming only.)
  4 Which predicts     5-fold cross-validated R2 of ridge regressions of ln(gross / budget) on budget decile + era,
                       then adding each representation, on the films that have all of them; repeated 5 times.

Outputs in analysis/genre/outputs/shares/. Numbers only.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_shares [--perms 10000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from analysis.genre.q1_genre import DATA, OUT as MAIN_OUT, SEED, holm, load, permuted_means, prepare, strata
from analysis.genre.exploratory import EXP_OUT

OUT = EXP_OUT / "shares"
KM_SEEDS = 10


def shares(path, f, keep):
    s = pd.read_csv(DATA / "features" / path).drop_duplicates("tconst").set_index("tconst")
    cols = [f"share_{g}" for g in keep]
    S = s.reindex(f["imdb_id"])[cols].to_numpy()
    return S, ~np.isnan(S).any(1)


def shuffle_test(y, W, f, perms, seed):
    obs = y @ W / W.sum(0)
    nl = permuted_means(y, W, strata(f, "budget"), perms, np.random.default_rng(seed))
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    p = (1 + (np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12).sum(0)) / (1 + perms)
    return obs, obs - mu, (obs - mu) / sd, holm(p)


def plot_clusters(f, k):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi
    V = np.load(DATA / "features" / "plot_vectors.npy")
    ids = pd.read_csv(DATA / "features" / "plot_vector_ids.csv")["tconst"]
    pos = {t: i for i, t in enumerate(ids)}
    rows = np.array([pos.get(t, -1) for t in f["imdb_id"]])
    have = rows >= 0
    X = V[rows[have]]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    fits = [KMeans(k, n_init=4, random_state=SEED + s).fit(X) for s in range(KM_SEEDS)]
    best = min(fits, key=lambda m: m.inertia_)
    stab = float(np.mean([nmi(best.labels_, m.labels_) for m in fits if m is not best]))
    lab = np.full(len(f), -1)
    lab[have] = best.labels_
    emb = np.full((len(f), V.shape[1]), np.nan)
    emb[have] = X
    return lab, have, stab, emb


def cv_r2(y, X, folds=5, reps=5):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    out = []
    for r in range(reps):
        pred = np.empty_like(y)
        for tr, te in KFold(folds, shuffle=True, random_state=SEED + r).split(X):
            m = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        out.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out)), float(np.std(out))


def main():
    from sklearn.metrics import normalized_mutual_info_score as nmi
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    y = f["ln_gross_over_budget"].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    G, has_g = shares("genre_shares.csv", f, keep)
    P, has_p = shares("plot_genre_shares.csv", f, keep)
    clus, has_c, stab, emb = plot_clusters(f, len(keep))
    summ = {"films": len(f), "with_tag_genome": int(has_g.sum()), "with_plot_shares": int(has_p.sum()),
            "with_plot_vector": int(has_c.sum()), "kmeans_k": len(keep), "kmeans_seeds": KM_SEEDS,
            "kmeans_mean_nmi_best_vs_other_seeds": round(stab, 3), "perms": a.perms}

    # 1 shares as weights, and 2 dominant genre, against 0/1 tags on the same films
    rows = []
    for name, S, has in (("tag_genome", G, has_g), ("plot", P, has_p)):
        fs, ys, Ms, Ss = f[has].reset_index(drop=True), y[has], M[has], S[has]
        Ss = Ss / np.where(Ss.sum(1, keepdims=True) > 0, Ss.sum(1, keepdims=True), 1)
        D = np.zeros_like(Ss)
        D[np.arange(len(Ss)), Ss.argmax(1)] = 1
        for rep, W, seed in (("imdb_tags", Ms, 1), (f"{name}_shares", Ss, 2), (f"{name}_dominant", D, 3)):
            ok = W.sum(0) > 0
            obs, ex, z, ph = shuffle_test(ys, W[:, ok], fs, a.perms, SEED + 900 + 10 * len(rows) + seed)
            for j, g in enumerate(np.array(keep)[ok]):
                rows.append({"films": name, "representation": rep, "genre": g, "weight": round(float(W[:, ok][:, j].sum()), 1),
                             "mean": round(obs[j], 4), "excess": round(ex[j], 4), "z": round(z[j], 2), "p_holm": round(ph[j], 5)})
    sh = pd.DataFrame(rows)
    sh.to_csv(OUT / "share_tests.csv", index=False)

    # 3 agreement between the ways of naming a film's genre, on films with all of them
    both = has_g & has_p & has_c
    Gd, Pd = G[both].argmax(1), P[both].argmax(1)
    Mi = np.where(M[both] > 0, G[both], -1).argmax(1)       # the IMDb genre with the largest Tag Genome share
    labels_ = {"tag_genome_dominant": Gd, "plot_dominant": Pd, "plot_cluster": clus[both]}
    agree = [{"a": x, "b": z, "nmi": round(nmi(labels_[x], labels_[z]), 3)}
             for i, x in enumerate(labels_) for z in list(labels_)[i + 1:]]
    rng = np.random.default_rng(SEED + 950)
    for x in ("tag_genome_dominant", "plot_dominant"):
        hit = M[both][np.arange(both.sum()), labels_[x]].mean()
        # null: the same dominant genres given to random films
        nl = [M[both][np.arange(both.sum()), rng.permutation(labels_[x])].mean() for _ in range(1000)]
        agree.append({"a": x, "b": "one of the film's IMDb genres", "share_of_films": round(float(hit), 3),
                      "null_mean": round(float(np.mean(nl)), 3)})
    pd.DataFrame(agree).to_csv(OUT / "agreement.csv", index=False)
    # what the clusters are, in IMDb terms
    cl = pd.DataFrame({"cluster": clus[both], "genre": np.array(keep)[Mi], "y": y[both]})
    desc = (cl.groupby("cluster").agg(films=("y", "size"), mean_ln_ratio=("y", "mean"))
            .join(cl.groupby("cluster")["genre"].agg(lambda s: ", ".join(f"{g} {c / len(s):.0%}" for g, c in s.value_counts().head(3).items())).rename("top_imdb_genres")))
    # cluster returns against the budget shuffle
    fc = f[both].reset_index(drop=True)
    C = np.eye(len(keep))[clus[both]]
    obs, ex, z, ph = shuffle_test(y[both], C, fc, a.perms, SEED + 990)
    desc["excess"], desc["z"], desc["p_holm"] = np.round(ex, 4), np.round(z, 2), np.round(ph, 5)
    desc.reset_index().to_csv(OUT / "plot_clusters.csv", index=False)

    # 4 cross-validated R2 on the films with every representation
    yb = y[both]
    base = pd.get_dummies(pd.DataFrame({"b": f["budget_bin"][both].astype(str), "e": f["era_bin"][both]}), dtype=float).to_numpy()
    reps = {"budget + era": base, "+ IMDb tags (0/1)": np.hstack([base, M[both]]),
            "+ Tag Genome shares": np.hstack([base, G[both]]), "+ plot shares": np.hstack([base, P[both]]),
            "+ plot clusters": np.hstack([base, C]), "+ plot embedding": np.hstack([base, emb[both]]),
            "+ IMDb tags + Tag Genome shares": np.hstack([base, M[both], G[both]])}
    r2 = []
    for name, X in reps.items():
        m, s = cv_r2(yb, X)
        r2.append({"model": name, "cv_r2": round(m, 4), "sd_over_repeats": round(s, 4), "features": X.shape[1]})
    r2 = pd.DataFrame(r2)
    r2["added_over_budget_era"] = (r2["cv_r2"] - r2.loc[0, "cv_r2"]).round(4)
    r2.to_csv(OUT / "cv_r2.csv", index=False)
    summ["films_with_all"] = int(both.sum())
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ))
    print(pd.DataFrame(agree).to_string(index=False))
    print(r2.to_string(index=False))
    print(desc.sort_values("excess").to_string())


if __name__ == "__main__":
    main()
