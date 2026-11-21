"""Q1 tied to the people (Q2's network) and the screenplays (Q3's text), using only what exists before release.
Added after the main results in q1_genre.py.

  A specialists   For every usable-money film, its team is its director(s), writers, producers and first three billed
                  actors (graph/credits.csv). Each person's earlier films are the films of the 12,673-film set released
                  in an earlier year (same-year films don't count, so nothing about the film itself or later films
                  leaks in). Per film:
                    genre_experience  the share of the team's earlier film credits that carry one of the film's
                                      genres (pooled over people, then averaged over the film's genres)
                    prior_credits     log(1 + the team's earlier film credits)
                    genre_breadth     entropy of the team's earlier credits over the 18 kept genres
                  Outcome: ln(gross / budget) net of the budget decile x era mean. OLS with genre dummies and a sequel
                  flag (Wikidata 'follows', features/franchise.csv), HC3, also with sequels left out; and a
                  shuffle of genre_experience within genre x budget decile cells (1000 times) for its p. Films whose
                  team has no earlier credit are left out of this part.
  B typicality    How close a script is to the usual script of its genres: the cosine of its embedding
                  (all-MiniLM-L6-v2, character labels removed) to the leave-one-out mean of the other usable films
                  sharing each of its genres, averaged over its genres, minus its cosine to the mean of all scripts.
                  Same outcome and controls (also without sequels), a shuffle within genre x budget decile, and quintiles.
  C Horror        The plot-word test of q1_words.py on the screenplays of the usable Horror films (description layer and
                  dialogue layer, the counts of Q3's preprocessing in features/script_w2v/script_counts.pkl), and the
                  post-hoc supernatural word set from the plot test, counted in the scripts.

Outputs in analysis/genre/outputs/scripts_network/. Numbers only; no script text leaves data/raw.
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_scripts_network [--perms 1000]
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from analysis.genre.q1_genre import DATA, OUT as MAIN_OUT, SEED, load, prepare
from analysis.genre.q1_network import ROLES, TOP_ACTORS
from analysis.genre.q1_words import SUPERNATURAL, log_odds_z

OUT = MAIN_OUT / "scripts_network"


def net_return(f):
    cell = f["budget_bin"].astype(str) + "|" + f["era_bin"].astype(str)
    return (f["ln_gross_over_budget"] - f.groupby(cell)["ln_gross_over_budget"].transform("mean")).to_numpy()


def sequel_flag(f):
    """1 when Wikidata says the film follows another (features/franchise.csv, 'follows'); the cleaner franchise flag."""
    fr = pd.read_csv(DATA / "features" / "franchise.csv").drop_duplicates("tconst").set_index("tconst")["follows"]
    return f["imdb_id"].map(fr).notna().to_numpy() & f["imdb_id"].map(fr).astype(str).str.len().gt(3).to_numpy()


def ols(y, X, names, f, M, keep):
    """Genre dummies and a sequel flag as controls; X standardised so coefficients are per SD."""
    import statsmodels.api as sm
    ctrl = pd.get_dummies(pd.DataFrame(M, columns=keep).astype(int).astype(str), drop_first=True, dtype=float)
    Z = np.column_stack([(X - X.mean(0)) / X.std(0), ctrl.to_numpy(), sequel_flag(f).astype(float)])
    fit = sm.OLS(y, sm.add_constant(Z)).fit(cov_type="HC3")
    ci = fit.conf_int()
    return [{"term": n, "coef_per_sd": round(fit.params[1 + i], 4), "lo": round(ci[1 + i, 0], 4), "hi": round(ci[1 + i, 1], 4),
             "p": float(f"{fit.pvalues[1 + i]:.3g}")} for i, n in enumerate(names)]


def shuffle_p(y, x, groups, M, perms, rng):
    """p for the coefficient of x (with genre dummies) when x is shuffled within groups."""
    G = np.column_stack([np.ones(len(y)), M])
    def coef(v):
        Z = np.column_stack([G, (v - v.mean()) / v.std()])
        return np.linalg.lstsq(Z, y, rcond=None)[0][-1]
    obs = coef(x)
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    nl = np.empty(perms)
    for p in range(perms):
        v = x.copy()
        for ix in idx:
            v[ix] = x[rng.permutation(ix)]
        nl[p] = coef(v)
    return obs, float((1 + (np.abs(nl) >= abs(obs)).sum()) / (1 + perms))


def specialists(films, genres, f, keep, M, perms):
    cr = pd.read_csv(DATA / "features" / "graph" / "credits.csv")
    cr = cr[cr["role"].isin(ROLES) | ((cr["role"] == "actor") & (cr["ordering"] <= TOP_ACTORS))]
    year = films.set_index("imdb_id")["year"]
    gi = {g: j for j, g in enumerate(keep)}
    fg = defaultdict(set)
    for t, g in zip(genres["imdb_id"], genres["genre"]):
        if g in gi:
            fg[t].add(gi[g])
    cr = cr.assign(year=cr["tconst"].map(year)).dropna(subset=["year"])
    by_person = {p: d.sort_values("year")[["tconst", "year"]].to_numpy() for p, d in cr.groupby("nconst")}
    team = cr.groupby("tconst")["nconst"].apply(lambda s: sorted(set(s)))
    rows = []
    for t, yr, mrow in zip(f["imdb_id"], f["year"], M):
        mine = np.flatnonzero(mrow)
        cnt, n = np.zeros(len(keep)), 0
        for p in team.get(t, []):
            for t2, y2 in by_person[p]:
                if y2 >= yr:
                    break
                n += 1
                for j in fg.get(t2, ()):
                    cnt[j] += 1
        if n == 0 or len(mine) == 0:
            rows.append((np.nan, 0.0, np.nan))
            continue
        q = cnt / cnt.sum() if cnt.sum() > 0 else cnt
        ent = float(-(q[q > 0] * np.log(q[q > 0])).sum())
        rows.append((float(np.mean(cnt[mine] / n)), np.log1p(n), ent))
    X = np.array(rows)
    ok = ~np.isnan(X).any(1)
    y = net_return(f)[ok]
    res = [dict(r, films="all, sequel flag controlled") for r in ols(y, X[ok], ["genre_experience", "prior_credits", "genre_breadth"], f[ok], M[ok], keep)]
    ns = ~sequel_flag(f[ok])
    res += [dict(r, films="sequels left out") for r in ols(y[ns], X[ok][ns], ["genre_experience", "prior_credits", "genre_breadth"], f[ok][ns], M[ok][ns], keep)]
    grp = pd.factorize(pd.Series(M[ok].argmax(1)).astype(str) + "|" + f["budget_bin"][ok].astype(str).to_numpy())[0]
    obs, p = shuffle_p(y, X[ok, 0], grp, M[ok], perms, np.random.default_rng(SEED + 1700))
    q = pd.qcut(X[ok, 0], 5, labels=False)
    quint = [{"quintile": k + 1, "films": int((q == k).sum()), "genre_experience_range": f"{X[ok, 0][q == k].min():.2f}-{X[ok, 0][q == k].max():.2f}",
              "mean_net_return": round(float(y[q == k].mean()), 4)} for k in range(5)]
    # per role: genre experience of directors only, and of the first three actors only
    summ = {"films": int(ok.sum()), "films_without_earlier_credits": int((~ok).sum()), "sequels": int((~ns).sum()),
            "shuffle_coef_genre_experience": round(float(obs), 4),
            "shuffle_p_genre_experience": round(p, 4), "perms": perms}
    return pd.DataFrame(res), pd.DataFrame(quint), summ


def typicality(f, keep, M, perms):
    V = np.load(DATA / "features" / "embeddings_hpc" / "script_vectors.npy")
    ids = pd.read_csv(DATA / "features" / "embeddings_hpc" / "ids.csv")["tconst"]
    pos = {t: i for i, t in enumerate(ids)}
    r = np.array([pos.get(t, -1) for t in f["imdb_id"]])
    ok = r >= 0
    X = V[r[ok]].astype(float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Mo = M[ok]
    tot = X.sum(0)
    sums = Mo.T @ X                      # genre x dim
    n = Mo.sum(0)
    typ = np.empty(len(X))
    for i in range(len(X)):
        gs = np.flatnonzero(Mo[i])
        sims = []
        for j in gs:
            if n[j] < 2:
                continue
            c = (sums[j] - X[i]) / (n[j] - 1)
            sims.append(X[i] @ c / np.linalg.norm(c))
        allc = (tot - X[i]) / (len(X) - 1)
        typ[i] = (np.mean(sims) if sims else np.nan) - X[i] @ allc / np.linalg.norm(allc)
    good = ~np.isnan(typ)
    fo = f[ok].reset_index(drop=True)[good].reset_index(drop=True)
    Mo, typ = Mo[good], typ[good]
    y = net_return(fo)
    res = [dict(r, films="all, sequel flag controlled") for r in ols(y, typ[:, None], ["script_typicality"], fo, Mo, keep)]
    ns = ~sequel_flag(fo)
    res += [dict(r, films="sequels left out") for r in ols(y[ns], typ[ns, None], ["script_typicality"], fo[ns], Mo[ns], keep)]
    grp = pd.factorize(pd.Series(Mo.argmax(1)).astype(str) + "|" + fo["budget_bin"].astype(str))[0]
    obs, p = shuffle_p(y, typ, grp, Mo, perms, np.random.default_rng(SEED + 1800))
    q = pd.qcut(typ, 5, labels=False)
    quint = [{"quintile": k + 1, "films": int((q == k).sum()), "mean_net_return": round(float(y[q == k].mean()), 4)} for k in range(5)]
    per_genre = []
    for j, g in enumerate(keep):
        m = Mo[:, j] == 1
        if m.sum() >= 100:
            rho = np.corrcoef(typ[m], y[m])[0, 1]
            per_genre.append({"genre": g, "films": int(m.sum()), "corr_typicality_net_return": round(float(rho), 3)})
    summ = {"films": len(fo), "shuffle_coef": round(float(obs), 4), "shuffle_p": round(p, 4), "perms": perms}
    return pd.DataFrame(res), pd.DataFrame(quint), pd.DataFrame(per_genre), summ


def horror_scripts(f, keep, M, perms):
    counts = pickle.loads((DATA / "features" / "script_w2v" / "script_counts.pkl").read_bytes())
    f = f.assign(net=net_return(f))
    h = f[(M[:, keep.index("Horror")] == 1) & f["imdb_id"].isin(counts)].reset_index(drop=True)
    q1, q3 = h["net"].quantile([.25, .75])
    lab = h[(h["net"] <= q1) | (h["net"] >= q3)].reset_index(drop=True)
    y = (lab["net"] >= q3).to_numpy().astype(int)
    rng = np.random.default_rng(SEED + 1900)
    words_out, lists, summ = [], [], {"films_with_script": len(h), "hits": int(y.sum()), "flops": int(len(y) - y.sum()), "perms": perms}
    for layer in ("description", "dialogue"):
        prior = Counter()
        for t in f["imdb_id"]:
            if t in counts:
                prior.update(counts[t][layer])
        df_ = Counter(w for t in lab["imdb_id"] for w in counts[t][layer])
        vocab = sorted(w for w, d in df_.items() if d >= 10)
        vi = {w: j for j, w in enumerate(vocab)}
        C = np.zeros((len(lab), len(vocab)))
        for i, t in enumerate(lab["imdb_id"]):
            for w, c in counts[t][layer].items():
                if w in vi:
                    C[i, vi[w]] = c
        alpha = np.array([prior[w] for w in vocab], float)
        zf = lambda yy: log_odds_z(C[yy == 1].sum(0), C[yy == 0].sum(0), alpha)
        obs = zf(y)
        grp = lab["budget_bin"].to_numpy()
        idx = [np.flatnonzero(grp == g) for g in np.unique(grp)]
        nl = np.empty((perms, len(vocab)))
        for p in range(perms):
            yp = y.copy()
            for ix in idx:
                yp[ix] = y[rng.permutation(ix)]
            nl[p] = zf(yp)
        zc = (obs - nl.mean(0)) / nl.std(0, ddof=1)
        zcn = (nl - nl.mean(0)) / nl.std(0, ddof=1)
        w = pd.DataFrame({"layer": layer, "word": vocab, "calibrated_z": zc.round(2),
                          "hit_films": (C[y == 1] > 0).sum(0), "flop_films": (C[y == 0] > 0).sum(0)})
        words_out.append(pd.concat([w.nlargest(15, "calibrated_z").assign(side="hit"), w.nsmallest(15, "calibrated_z").assign(side="flop")]))
        summ[f"{layer}_vocab"] = len(vocab)
        summ[f"{layer}_familywise_p"] = round(float((1 + (np.abs(zcn).max(1) >= np.abs(zc).max()).sum()) / (1 + perms)), 4)
        for side, sg in (("hit", 1), ("flop", -1)):
            o, n_ = np.sort(sg * zc)[::-1], np.sort(sg * zcn, axis=1)[:, ::-1]
            for k in (5, 10, 20):
                lists.append({"layer": layer, "side": side, "k": k, "p": round(float((1 + (n_[:, k - 1] >= o[k - 1]).sum()) / (1 + perms)), 4)})
        sup = set(SUPERNATURAL)
        summ[f"{layer}_supernatural_words_in_vocab"] = sorted(sup & set(vocab))
    # supernatural set counted in the scripts (description + dialogue), on all Horror films with a script
    sup_film = np.array([any(counts[t][l].get(w, 0) > 0 for l in ("description", "dialogue") for w in SUPERNATURAL) for t in h["imdb_id"]])
    net = h["net"].to_numpy()
    d = net[sup_film].mean() - net[~sup_film].mean()
    nl = np.array([(lambda m: net[m].mean() - net[~m].mean())(rng.permutation(sup_film)) for _ in range(5000)])
    summ["supernatural_in_script"] = {"films": int(sup_film.sum()), "share": round(float(sup_film.mean()), 3),
                                      "net_supernatural": round(float(net[sup_film].mean()), 3), "net_other": round(float(net[~sup_film].mean()), 3),
                                      "difference": round(float(d), 3), "p": round(float((1 + (np.abs(nl) >= abs(d)).sum()) / 5001), 4)}
    return pd.concat(words_out), pd.DataFrame(lists), summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=1000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {}
    r, q, s = specialists(films, genres, f, keep, M, a.perms)
    r.to_csv(OUT / "specialists_ols.csv", index=False)
    q.to_csv(OUT / "specialists_quintiles.csv", index=False)
    summ["specialists"] = s
    print(r.to_string(index=False), q.to_string(index=False), s, sep="\n", flush=True)
    r, q, pg, s = typicality(f, keep, M, a.perms)
    r.to_csv(OUT / "typicality_ols.csv", index=False)
    q.to_csv(OUT / "typicality_quintiles.csv", index=False)
    pg.to_csv(OUT / "typicality_per_genre.csv", index=False)
    summ["typicality"] = s
    print(r.to_string(index=False), q.to_string(index=False), pg.to_string(index=False), s, sep="\n", flush=True)
    w, l, s = horror_scripts(f, keep, M, a.perms)
    w.to_csv(OUT / "horror_script_words.csv", index=False)
    l.to_csv(OUT / "horror_script_list_null.csv", index=False)
    summ["horror_scripts"] = s
    print(w[["layer", "side", "word", "calibrated_z", "hit_films", "flop_films"]].to_string(index=False), l.to_string(index=False), json.dumps(s, indent=1), sep="\n")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
