"""Fixes for the weaknesses of the Tag Genome genre shares (features/genre_shares.csv), found after the follow-ups in
q1_share_followups.py. Every choice below is made from agreement with IMDb's labels, never from returns, and is fixed
before the return tests in this script run.

  1 mapping    each genre was scored by one Tag Genome tag. For each genre a fixed set of candidate tags (and their
               mean) is scored by AUC against IMDb's label; the best one is kept.
  2 power      shares = relevance ** p / row total. p is picked from P_GRID by the cross-validated log loss of
               predicting each IMDb label from its share (one logistic regression per genre), not by returns.
  3 split      raw relevance mixes how much a film is tagged at all with what kind of film it is. log(total
               relevance over the kept genres) enters as its own covariate, and the shares stay pure composition.
  4 reception  IMDb rating and ln(IMDb votes) go into the base model, so the shares are judged on what they add beyond
               reception and audience size; Rotten Tomatoes joins them on the films that have it. (Residualising the
               shares on rating and votes and leaving those out of the model is no fix: a linear model on the residual
               columns can rebuild rating and votes from them. By Frisch-Waugh-Lovell, holding them in the model is the
               same as using the residuals with them held fixed.)
  5 coverage   the Tag Genome stops in 2014 and favours much-tagged films. Each film with scores is weighted by 1 /
               (the share of usable-money films with scores in its decade x vote tier x budget decile cell), and the
               budget-netted genre means with 0/1 tags are compared on the full set, the subset, and the reweighted
               subset.
  6 spread     the effective number of genres (1 / sum of squared shares) against the budget-netted return,
               holding log total, rating and ln(votes) fixed (OLS, HC3).
  7 scripts    pre-release genre shares: a ridge regression from each screenplay's embedding (all-MiniLM-L6-v2,
               features/embeddings_hpc/script_vectors.npy, character labels removed) to the chosen shares, trained on
               films with both, out-of-fold for those films and fitted on all of them for the rest. A Word2Vec
               description + dialogue vector (window 5, seed 1, features/script_w2v/) is the second representation.
               A script exists before release and holds no ratings, so these shares can't carry reception.

Return comparisons use 5-fold CV R2 (5 repeats) of ln(gross / budget) on budget decile + era plus each genre
representation, on the same films. Outputs in analysis/genre/outputs/share_fixes/. Numbers only.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_share_fixes [--perms 10000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from analysis.genre.q1_genre import DATA, SEED, holm, load, permuted_means, prepare, strata
from analysis.genre.exploratory import EXP_OUT
from analysis.genre.exploratory.q1_shares import cv_r2

OUT = EXP_OUT / "share_fixes"
P_GRID = (0.5, 1, 1.5, 2, 3, 4)
CANDIDATES = {
    "Action": ["action"], "Adventure": ["adventure"], "Animation": ["animation", "animated"],
    "Biography": ["biographical", "biography", "true story"], "Comedy": ["comedy", "funny"], "Crime": ["crime"],
    "Drama": ["drama", "dramatic"], "Family": ["kids and family", "family", "kids", "children"],
    "Fantasy": ["fantasy"], "History": ["history", "historical"], "Horror": ["horror", "scary", "creepy"],
    "Music": ["music"], "Mystery": ["mystery"], "Romance": ["romance", "romantic"],
    "Sci-Fi": ["sci-fi", "science fiction"], "Sport": ["sports"], "Thriller": ["thriller", "suspense"], "War": ["war"],
}
CURRENT = {"Action": "action", "Adventure": "adventure", "Animation": "animation", "Biography": "biographical",
           "Comedy": "comedy", "Crime": "crime", "Drama": "drama", "Family": "kids and family", "Fantasy": "fantasy",
           "History": "history", "Horror": "horror", "Music": "music", "Mystery": "mystery", "Romance": "romance",
           "Sci-Fi": "sci-fi", "Sport": "sports", "Thriller": "thriller", "War": "war"}
HAND_SAMPLE = 40


def base_matrix(f):
    return pd.get_dummies(pd.DataFrame({"b": f["budget_bin"].astype(str), "e": f["era_bin"]}), dtype=float).to_numpy()


def relevance(f):
    R = np.load(DATA / "features" / "tag_relevance.npy")
    ids = pd.read_csv(DATA / "features" / "tag_relevance_ids.csv")["tconst"]
    tags = pd.read_csv(DATA / "features" / "tag_relevance_tags.csv")["tag"].tolist()
    pos = {t: i for i, t in enumerate(ids)}
    rows = np.array([pos.get(t, -1) for t in f["imdb_id"]])
    return np.clip(R, 0, 1), rows, {t: j for j, t in enumerate(tags)}


def mapping(R, rows, tix, keep, M):
    from sklearn.metrics import roc_auc_score
    has = rows >= 0
    out, cols = [], []
    for j, g in enumerate(keep):
        y = M[has, j]
        cand = {t: R[rows[has], tix[t]] for t in CANDIDATES[g]}
        if len(cand) > 1:
            cand["mean of " + ", ".join(CANDIDATES[g])] = np.mean(list(cand.values()), axis=0)
        aucs = {k: roc_auc_score(y, v) for k, v in cand.items()}
        best = max(aucs, key=aucs.get)
        out += [{"genre": g, "candidate": k, "auc": round(a, 4), "current": k == CURRENT[g], "chosen": k == best}
                for k, a in aucs.items()]
        cols.append(cand[best])
    return pd.DataFrame(out), np.column_stack(cols)


def to_shares(Rg, p):
    X = Rg ** p
    return X / X.sum(1, keepdims=True)


def pick_power(Rg, M):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    rows = []
    for p in P_GRID:
        S = to_shares(Rg, p)
        loss = np.mean([-cross_val_score(LogisticRegression(), S[:, [j]], M[:, j], cv=5, scoring="neg_log_loss").mean()
                        for j in range(M.shape[1])])
        rows.append({"power": p, "mean_cv_log_loss_imdb_labels": round(float(loss), 5)})
    d = pd.DataFrame(rows)
    return d, float(d.loc[d["mean_cv_log_loss_imdb_labels"].idxmin(), "power"])


def r2_table(y, base, reps, label):
    r0 = cv_r2(y, base)[0]
    return [{"films": label, "model": k, "cv_r2": round(cv_r2(y, np.hstack([base, X]))[0], 4),
             "added": round(cv_r2(y, np.hstack([base, X]))[0] - r0, 4)} for k, X in reps.items()]


def genre_coefs(y, f, X, genres):
    """OLS of y on the genre columns (the first len(genres) columns of X), any extra covariates in X, and budget decile
    and era dummies; HC3 errors. Shares add up to 1, so for shares the first genre is the reference (its coefficient
    is 0) and the intervals are against it; the coefficients are then centred on their mean over the genres."""
    import statsmodels.api as sm
    ctrl = pd.get_dummies(pd.DataFrame({"b": f["budget_bin"].astype(str), "e": f["era_bin"]}), drop_first=True, dtype=float).to_numpy()
    k = len(genres)
    sums_to_one = np.allclose(X[:, :k].sum(1), 1)
    Xg = np.hstack([X[:, 1:k] if sums_to_one else X[:, :k], X[:, k:], ctrl])
    fit = sm.OLS(y, sm.add_constant(Xg)).fit(cov_type="HC3")
    kg = k - 1 if sums_to_one else k
    b, ci = fit.params[1:1 + kg], fit.conf_int()[1:1 + kg]
    if sums_to_one:
        b, ci = np.r_[0.0, b], np.vstack([[0.0, 0.0], ci])
    m = b.mean()
    return pd.DataFrame({"genre": genres, "coef_centred": (b - m).round(4), "lo": (ci[:, 0] - m).round(4),
                         "hi": (ci[:, 1] - m).round(4), "reference": genres[0] if sums_to_one else ""})


def script_shares(f_all, S_tg, has_tg, keep):
    """Out-of-fold ridge predictions of the Tag Genome shares from script vectors; returns {name: (n_films x k) or NaN}."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    reps = {}
    mini = np.load(DATA / "features" / "embeddings_hpc" / "script_vectors.npy")
    mids = pd.read_csv(DATA / "features" / "embeddings_hpc" / "ids.csv")["tconst"]
    z = np.load(DATA / "features" / "script_w2v" / "vectors_w5_s1.npz", allow_pickle=True)
    w2v = np.hstack([z["description_tfidf_mean"], z["dialogue_tfidf_mean"]])
    for name, V, ids in (("script_minilm", mini, mids), ("script_word2vec", w2v, pd.Series(z["ids"]))):
        pos = {t: i for i, t in enumerate(ids)}
        r = np.array([pos.get(t, -1) for t in f_all["imdb_id"]])
        X = np.full((len(f_all), V.shape[1]), np.nan)
        X[r >= 0] = V[r[r >= 0]]
        sd = np.nanstd(X, 0)
        X = (X - np.nanmean(X, 0)) / np.where(sd > 0, sd, 1)
        X[r >= 0] = np.nan_to_num(X[r >= 0])
        train = (r >= 0) & has_tg
        pred = np.full((len(f_all), len(keep)), np.nan)
        alphas = np.logspace(0, 5, 21)
        tr_idx = np.flatnonzero(train)
        for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(tr_idx)):
            m = RidgeCV(alphas=alphas).fit(X[tr_idx[a]], S_tg[tr_idx[a]])
            pred[tr_idx[b]] = m.predict(X[tr_idx[b]])
        rest = (r >= 0) & ~has_tg
        if rest.any():
            pred[rest] = RidgeCV(alphas=alphas).fit(X[train], S_tg[train]).predict(X[rest])
        P = np.clip(pred, 1e-6, None)
        reps[name] = P / P.sum(1, keepdims=True)
    return reps


def main():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    import statsmodels.api as sm
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    films, genres = load()
    f_all, keep, M_all = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)
    R, rows, tix = relevance(f_all)
    mf = pd.read_csv(DATA / "features" / "movies_features_10k.csv", usecols=["tconst", "votes", "rating"]).drop_duplicates("tconst").set_index("tconst")
    rt = pd.read_csv(DATA / "features" / "wiki_infobox_10k.csv", usecols=["tconst", "rt_pct"]).drop_duplicates("tconst").set_index("tconst")["rt_pct"]
    f_all["rating"] = f_all["imdb_id"].map(mf["rating"])
    f_all["ln_votes"] = np.log(f_all["imdb_id"].map(mf["votes"]))
    f_all["rt"] = f_all["imdb_id"].map(rt)
    has = rows >= 0
    summ = {"films": len(f_all), "films_with_tag_genome": int(has.sum())}

    # 1 mapping and 2 power, on the films with scores
    mp, Rg_has = mapping(R, rows, tix, keep, M_all)
    mp.to_csv(OUT / "mapping.csv", index=False)
    pw, p_star = pick_power(Rg_has, M_all[has])
    pw.to_csv(OUT / "power.csv", index=False)
    summ["power_chosen"] = p_star
    summ["mapping_changed"] = mp[mp["chosen"] & ~mp["current"]]["genre"].tolist()
    f = f_all[has].reset_index(drop=True)
    M = M_all[has]
    y = f["ln_gross_over_budget"].to_numpy()
    S = to_shares(Rg_has, p_star)
    logT = np.log(Rg_has.sum(1))[:, None]
    base = base_matrix(f)

    # 3 split and 4 reception, CV R2 on the same films
    rec = f[["rating", "ln_votes"]].to_numpy()
    reps = {"IMDb tags": M, "shares (old mapping, p=2)": to_shares(np.column_stack([R[rows[has], tix[CURRENT[g]]] for g in keep]), 2),
            "shares (chosen)": S, "log total only": logT, "shares (chosen) + log total": np.hstack([S, logT]),
            "raw relevance": Rg_has}
    for p in P_GRID:
        reps[f"shares p={p}"] = to_shares(Rg_has, p)
    r2 = r2_table(y, base, reps, "tag_genome_films")
    # reception held fixed: rating and ln(votes) in the base model. (Residualising the shares on them and leaving them
    # out of the base is not a fix: a linear model on the residuals can rebuild rating and votes from the 18 columns.)
    r2 += r2_table(y, np.hstack([base, rec]), {k: reps[k] for k in ("IMDb tags", "shares (chosen)", "shares (chosen) + log total",
                                                                    "raw relevance")}, "tag_genome_films, rating and votes in base")
    ok = f["rt"].notna().to_numpy()
    r2 += r2_table(y[ok], np.hstack([base[ok], rec[ok], f["rt"].to_numpy()[ok, None]]),
                   {"IMDb tags": M[ok], "shares (chosen)": S[ok]}, "films with Rotten Tomatoes, rating, votes and RT in base")
    for tgt in ("rating", "ln_votes"):
        r2 += [dict(r, films=f"target {tgt}") for r in r2_table(f[tgt].to_numpy(), base, {k: reps[k] for k in ("IMDb tags", "shares (chosen)", "log total only")}, "")]
    pd.DataFrame(r2).to_csv(OUT / "cv_r2.csv", index=False)

    # genre effects: tags vs shares vs reception-free shares (centred OLS coefficients next to budget + era)
    co = []
    for name, X in (("IMDb tags", M), ("IMDb tags, rating and votes fixed", np.hstack([M, rec])),
                    ("shares + log total", np.hstack([S, logT])),
                    ("shares + log total, rating and votes fixed", np.hstack([S, logT, rec]))):
        co.append(genre_coefs(y, f, X, keep).assign(representation=name))
    co = pd.concat(co)
    co.to_csv(OUT / "genre_effects.csv", index=False)
    piv = co.pivot(index="genre", columns="representation", values="coef_centred")
    summ["spearman_effects"] = {f"{a_} ~ {b_}": round(float(spearmanr(piv[a_], piv[b_])[0]), 3)
                                for i, a_ in enumerate(piv.columns) for b_ in piv.columns[i + 1:]}

    # 5 coverage: budget-netted genre means with tags, full vs subset vs reweighted subset
    net_all = f_all["ln_gross_over_budget"] - f_all.groupby("budget_bin")["ln_gross_over_budget"].transform("mean")
    cell = f_all["decade"].astype(str) + "|" + f_all["tier"] + "|" + f_all["budget_bin"].astype(str)
    p_cell = pd.Series(has).groupby(cell.to_numpy()).transform("mean").to_numpy()
    w = np.where(has, 1 / np.maximum(p_cell, 1e-9), 0)
    cov = pd.DataFrame({"genre": keep,
                        "full_set": (net_all.to_numpy() @ M_all) / M_all.sum(0),
                        "subset": (net_all.to_numpy()[has] @ M) / M.sum(0),
                        "subset_reweighted": ((net_all.to_numpy() * w) @ M_all) / (w @ M_all)}).round(4)
    cov.to_csv(OUT / "coverage.csv", index=False)
    summ["coverage"] = {"cells_without_scored_films_share_of_full": round(float((p_cell == 0).mean()), 3),
                        "corr_full_vs_subset": round(float(np.corrcoef(cov["full_set"], cov["subset"])[0, 1]), 3),
                        "corr_full_vs_reweighted": round(float(np.corrcoef(cov["full_set"], cov["subset_reweighted"])[0, 1]), 3),
                        "mean_abs_gap_subset": round(float((cov["full_set"] - cov["subset"]).abs().mean()), 4),
                        "mean_abs_gap_reweighted": round(float((cov["full_set"] - cov["subset_reweighted"]).abs().mean()), 4)}

    # 6 spread, holding log total and reception fixed
    eff = 1 / (S ** 2).sum(1)
    net = y - pd.Series(y).groupby(f["budget_bin"].to_numpy()).transform("mean").to_numpy()
    sp = []
    for name, Z in (("effective genres alone", np.column_stack([eff])),
                    ("+ log total", np.column_stack([eff, logT[:, 0]])),
                    ("+ log total, rating, ln votes", np.column_stack([eff, logT[:, 0], rec])),
                    ("IMDb number of genres, + rating, ln votes", np.column_stack([M.sum(1), rec]))):
        fit = sm.OLS(net, sm.add_constant((Z - Z.mean(0)) / Z.std(0))).fit(cov_type="HC3")
        ci = fit.conf_int()[1]
        sp.append({"model": name, "coef_per_sd": round(fit.params[1], 4), "lo": round(ci[0], 4), "hi": round(ci[1], 4), "p": float(f"{fit.pvalues[1]:.3g}")})
    pd.DataFrame(sp).to_csv(OUT / "spread.csv", index=False)

    # 7 script-predicted shares, pre-release, on the films with a script
    S_all = np.full((len(f_all), len(keep)), np.nan)
    S_all[has] = S
    sc = script_shares(f_all, S_all, has, keep)
    val, r2s = [], []
    for name, P in sc.items():
        got = ~np.isnan(P).any(1)
        both = got & has
        val.append({"representation": name, "films_with_script": int(got.sum()), "trained_on": int(both.sum()),
                    "mean_auc_vs_imdb_labels": round(float(np.mean([roc_auc_score(M_all[got, j], P[got, j]) for j in range(len(keep))])), 3),
                    "median_corr_with_tag_genome_share_oof": round(float(np.median([np.corrcoef(P[both, j], S_all[both, j])[0, 1] for j in range(len(keep))])), 3),
                    "dominant_genre_in_imdb_tags": round(float(M_all[got][np.arange(got.sum()), P[got].argmax(1)].mean()), 3)})
    common = f_all["in_common_set"].to_numpy() & ~np.isnan(sc["script_minilm"]).any(1) & ~np.isnan(sc["script_word2vec"]).any(1)
    fc = f_all[common].reset_index(drop=True)
    yc, bc, Mc = fc["ln_gross_over_budget"].to_numpy(), base_matrix(fc), M_all[common]
    recc = fc[["rating", "ln_votes"]].to_numpy()
    repc = {"IMDb tags": Mc, "script shares (MiniLM)": sc["script_minilm"][common], "script shares (Word2Vec)": sc["script_word2vec"][common],
            "IMDb tags + script shares (MiniLM)": np.hstack([Mc, sc["script_minilm"][common]])}
    r2s += r2_table(yc, bc, repc, "common set")
    r2s += r2_table(yc, np.hstack([bc, recc]), repc, "common set, rating and votes in base")
    for tgt in ("rating", "ln_votes"):
        r2s += [dict(r, films=f"common set, target {tgt}") for r in r2_table(fc[tgt].to_numpy(), bc, repc, "")]
    pd.DataFrame(val).to_csv(OUT / "script_shares_validation.csv", index=False)
    pd.DataFrame(r2s).to_csv(OUT / "script_shares_cv_r2.csv", index=False)
    # budget-matched shuffle on the common set: tags vs script shares as weights
    st = []
    for k, (name, W) in enumerate((("IMDb tags", Mc), ("script shares (MiniLM)", sc["script_minilm"][common]))):
        ok_ = W.sum(0) > 0
        obs = yc @ W[:, ok_] / W[:, ok_].sum(0)
        nl = permuted_means(yc, W[:, ok_], strata(fc, "budget"), a.perms, np.random.default_rng(SEED + 1500 + k))
        mu, sd = nl.mean(0), nl.std(0, ddof=1)
        pv = (1 + (np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12).sum(0)) / (1 + a.perms)
        st += [{"representation": name, "genre": g, "weight": round(float(W[:, ok_][:, j].sum()), 1), "excess": round(obs[j] - mu[j], 4),
                "z": round((obs[j] - mu[j]) / sd[j], 2), "p_holm": round(h, 5)} for j, (g, h) in enumerate(zip(np.array(keep)[ok_], holm(pv)))]
    pd.DataFrame(st).to_csv(OUT / "script_shares_shuffle.csv", index=False)
    summ["common_set_films"] = int(common.sum())

    # hand-check sheet: films whose dominant share is none of their IMDb genres (filled in by people, not here)
    dom = S.argmax(1)
    dis = np.flatnonzero(M[np.arange(len(M)), dom] == 0)
    summ["films_dominant_not_in_imdb"] = int(len(dis))
    pick = np.random.default_rng(SEED + 1600).choice(dis, min(HAND_SAMPLE, len(dis)), replace=False)
    sheet = pd.DataFrame({"imdb_id": f["imdb_id"].to_numpy()[pick], "title": f["title"].to_numpy()[pick], "year": f["year"].to_numpy()[pick],
                          "imdb_genres": f["genres"].to_numpy()[pick],
                          "tag_genome_dominant": np.array(keep)[dom[pick]],
                          "tag_genome_top3": [", ".join(f"{keep[j]} {S[i, j]:.2f}" for j in np.argsort(-S[i])[:3]) for i in pick],
                          "which_is_right": "", "notes": ""})
    sheet.sort_values("imdb_id").to_csv(OUT / "hand_check_sheet.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(mp[mp["chosen"] | mp["current"]].to_string(index=False))
    print(pw.to_string(index=False))
    print(pd.DataFrame(r2).to_string(index=False))
    print(cov.to_string(index=False))
    print(pd.DataFrame(sp).to_string(index=False))
    print(pd.DataFrame(val).to_string(index=False))
    print(pd.DataFrame(r2s).to_string(index=False))
    print(co.pivot(index="genre", columns="representation", values="coef_centred").to_string())
    print(pd.DataFrame(st).pivot(index="genre", columns="representation", values=["excess", "p_holm"]).to_string())


if __name__ == "__main__":
    main()
