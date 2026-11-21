"""Q1 with TMDB as a second source (data/raw/tmdb/tmdb_project_movies.csv, the Kaggle TMDB file; never committed).
TMDB's genres, money, US certification and keywords are assigned by different people from IMDb's and Wikipedia's,
so they test whether the genre result depends on our sources, and they add two questions of their own.

Settings, fixed before any of these results was looked at:
  A genres     TMDB genre names mapped to IMDb's (Science Fiction -> Sci-Fi; the rest share names). TMDB has no
               Biography, Sport or Musical, so 15 of the 18 kept genres can be compared.
                 agreement   per genre: share of films where the two agree, Cohen's kappa, P(TMDB | IMDb), P(IMDb | TMDB)
                 main test   the budget-decile shuffle of q1_genre.py (10,000 shuffles, Holm) on the usable-money films,
                             with (a) IMDb genres, (b) TMDB genres, (c) consensus genres (a film counts in a genre only
                             when both sources give it)
  B money      TMDB budget and revenue, a TMDB 0 read as "no figure", both at least the shared rule's minimum, release
               years inside the shared rule's range. TMDB has no currency or distributor fields, so its figures are
               taken as US dollars and the streaming check can't be applied.
                 same films  the main test with ln(TMDB revenue / TMDB budget) on the usable-money films TMDB has
                             figures for
                 extended    the films Wikipedia gave no budget or gross for ("no budget or gross parsed") that have
                             clean TMDB money; films the shared rule drops on purpose (direct-to-video, rentals,
                             non-dollar, streaming) stay out. The main test on those alone, and on the union
                             (Wikipedia figures where usable, TMDB's otherwise)
  C rating     TMDB's US certification (G, PG, PG-13, R; NR, NC-17 and missing pooled as "other"). The return net of
               budget decile and era on genre dummies, with and without rating dummies (OLS, HC3): does the rating
               take over the genre effects? And each rating group against the budget-decile shuffle.
  D subgenres  inside IMDb-tagged Horror, subgenres from TMDB keywords, with keyword sets written down before looking:
                 supernatural  supernatural, demon, demonic possession, possession, ghost, haunted house, haunting,
                               exorcism, paranormal, poltergeist, witch, curse, occult, evil spirit
                 slasher       slasher, serial killer, masked killer, stalker
                 found footage found footage
                 creature      monster, creature, creature feature, alien, giant monster, zombie, vampire, werewolf
               Each subgenre's mean return net of budget decile and era against shuffles of that return among the
               Horror films within budget deciles. The supernatural set is a direct, independent test of the
               plot-word finding in exploratory/q1_words.py, which suggested it.

Outputs in analysis/genre/outputs/tmdb/: aggregate numbers only, no TMDB per-film figures (the TMDB file stays in
data/raw). Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_tmdb [--perms 10000]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from analysis.clean_rule import MAX_YEAR_EXCLUSIVE, MIN_BUDGET_USD, MIN_GROSS_USD, MIN_YEAR_INCLUSIVE
from analysis.genre.q1_genre import DATA, OUT as MAIN_OUT, SEED, holm, load, permuted_means, prepare, strata

OUT = MAIN_OUT / "tmdb"
NAME = {"Science Fiction": "Sci-Fi"}
RATINGS = ("G", "PG", "PG-13", "R")
SUBGENRES = {
    "supernatural": {"supernatural", "demon", "demonic possession", "possession", "ghost", "haunted house", "haunting",
                     "exorcism", "paranormal", "poltergeist", "witch", "curse", "occult", "evil spirit"},
    "slasher": {"slasher", "serial killer", "masked killer", "stalker"},
    "found footage": {"found footage"},
    "creature": {"monster", "creature", "creature feature", "alien", "giant monster", "zombie", "vampire", "werewolf"},
}


def tmdb():
    t = pd.read_csv(DATA / "tmdb" / "tmdb_project_movies.csv",
                    usecols=["imdb_id", "genres", "keywords", "certification_us", "budget", "revenue", "release_date"])
    return t.drop_duplicates("imdb_id").set_index("imdb_id")


def tmdb_genres(t, ids, keep):
    gi = {g: j for j, g in enumerate(keep)}
    G = np.zeros((len(ids), len(keep)))
    for i, s in enumerate(t["genres"].reindex(ids)):
        if isinstance(s, str):
            for g in s.split(","):
                g = NAME.get(g.strip(), g.strip())
                if g in gi:
                    G[i, gi[g]] = 1
    return G


def shuffle_test(f, y, W, perms, seed, label, keep):
    ok = W.sum(0) > 0
    obs = y @ W[:, ok] / W[:, ok].sum(0)
    nl = permuted_means(y, W[:, ok], strata(f, "budget"), perms, np.random.default_rng(seed))
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    p = (1 + (np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12).sum(0)) / (1 + perms)
    return [{"run": label, "genre": g, "n_films": int(W[:, ok][:, j].sum()), "mean": round(obs[j], 4),
             "excess": round(obs[j] - mu[j], 4), "z": round((obs[j] - mu[j]) / sd[j], 2), "p_holm": round(h, 5)}
            for j, (g, h) in enumerate(zip(np.array(keep)[ok], holm(p)))]


def rebin(f):
    f = f.copy()
    f["budget_bin"] = pd.qcut(np.log(f["budget"]).rank(method="first"), 10, labels=False)
    return f


def kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def net_return(f):
    cell = f["budget_bin"].astype(str) + "|" + f["era_bin"].astype(str)
    return (f["ln_gross_over_budget"] - f.groupby(cell)["ln_gross_over_budget"].transform("mean")).to_numpy()


def main():
    import statsmodels.api as sm
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    t = tmdb()
    OUT.mkdir(parents=True, exist_ok=True)
    shared = [g for g in keep if g not in ("Biography", "Sport", "Musical")]
    js = [keep.index(g) for g in shared]
    summ = {"usable_money_films": len(f), "with_tmdb_row": int(f["imdb_id"].isin(t.index).sum()), "shared_genres": shared,
            "perms": a.perms}

    # A genres
    T = tmdb_genres(t, f["imdb_id"], keep)
    has_t = T.sum(1) > 0
    agree = []
    for j, g in zip(js, shared):
        i_, t_ = M[has_t, j], T[has_t, j]
        agree.append({"genre": g, "imdb_films": int(i_.sum()), "tmdb_films": int(t_.sum()), "both": int((i_ * t_).sum()),
                      "agreement": round(float((i_ == t_).mean()), 4), "kappa": round(float(kappa(i_, t_)), 3),
                      "p_tmdb_given_imdb": round(float((i_ * t_).sum() / i_.sum()), 3),
                      "p_imdb_given_tmdb": round(float((i_ * t_).sum() / t_.sum()), 3) if t_.sum() else np.nan})
    pd.DataFrame(agree).to_csv(OUT / "genre_agreement.csv", index=False)
    fA, yA = f[has_t].reset_index(drop=True), f["ln_gross_over_budget"].to_numpy()[has_t]
    rows = []
    for k, (label, W) in enumerate((("IMDb genres", M[has_t][:, js]), ("TMDB genres", T[has_t][:, js]),
                                    ("consensus genres", (M[has_t] * T[has_t])[:, js]))):
        rows += shuffle_test(fA, yA, W, a.perms, SEED + 3000 + k, label, shared)
    summ["genre_test_films"] = int(has_t.sum())

    # B money
    tt = t.reindex(films["imdb_id"])
    yr = pd.to_numeric(films["year"], errors="coerce").to_numpy()
    tb, tr = tt["budget"].to_numpy(float), tt["revenue"].to_numpy(float)
    tmdb_ok = (tb >= MIN_BUDGET_USD) & (tr >= MIN_GROSS_USD) & (yr >= MIN_YEAR_INCLUSIVE) & (yr < MAX_YEAR_EXCLUSIVE)
    fm = films.assign(tmdb_budget=tb, tmdb_ln=np.log(np.where(tmdb_ok, tr, np.nan) / np.where(tmdb_ok, tb, np.nan)), tmdb_ok=tmdb_ok)
    # same films: usable-money films with clean TMDB money, TMDB's ratio
    same = f.merge(fm[["imdb_id", "tmdb_ok", "tmdb_ln", "tmdb_budget"]], on="imdb_id")
    sm_ = same["tmdb_ok"].to_numpy()
    fs = same[sm_].reset_index(drop=True)
    fs_t = rebin(fs.assign(budget=fs["tmdb_budget"]))
    rows += shuffle_test(fs, fs["ln_gross_over_budget"].to_numpy(), M[sm_], a.perms, SEED + 3010, "Wikipedia money, films with both", keep)
    rows += shuffle_test(fs_t, fs["tmdb_ln"].to_numpy(), M[sm_], a.perms, SEED + 3011, "TMDB money, films with both", keep)
    both_r = np.corrcoef(fs["ln_gross_over_budget"], fs["tmdb_ln"])[0, 1]
    # extended: films with clean TMDB money that the usable-money set leaves out
    # only films Wikipedia gave no figures for; films the shared rule drops on purpose (direct-to-video, rentals,
    # non-dollar, streaming, outside the years) stay out
    out_set = fm[(fm["money_exclude_reason"] == "no budget or gross parsed") & fm["tmdb_ok"]].copy()
    ex, _, Mx = prepare(films.assign(usable_money=films["imdb_id"].isin(set(out_set["imdb_id"])),
                                     ln_gross_over_budget=films["imdb_id"].map(fm.set_index("imdb_id")["tmdb_ln"]),
                                     budget=films["imdb_id"].map(fm.set_index("imdb_id")["tmdb_budget"])),
                        genres, "usable_money", keep=keep)
    rows += shuffle_test(ex, ex["ln_gross_over_budget"].to_numpy(), Mx, a.perms, SEED + 3012, "TMDB money, films Wikipedia lacks", keep)
    union_money = films.assign(
        usable_money=films["imdb_id"].isin(set(f["imdb_id"]) | set(out_set["imdb_id"])),
        ln_gross_over_budget=np.where(films["imdb_id"].isin(set(f["imdb_id"])), films["ln_gross_over_budget"],
                                      films["imdb_id"].map(fm.set_index("imdb_id")["tmdb_ln"])),
        budget=np.where(films["imdb_id"].isin(set(f["imdb_id"])), films["budget"], films["imdb_id"].map(fm.set_index("imdb_id")["tmdb_budget"])))
    fu, _, Mu = prepare(union_money, genres, "usable_money", keep=keep)
    rows += shuffle_test(fu, fu["ln_gross_over_budget"].to_numpy(), Mu, a.perms, SEED + 3013, "union: Wikipedia, else TMDB", keep)
    pd.DataFrame(rows).to_csv(OUT / "genre_tests.csv", index=False)
    summ["money"] = {"usable_films_with_clean_tmdb_money": int(sm_.sum()), "corr_ln_ratio_wikipedia_tmdb": round(float(both_r), 3),
                     "films_wikipedia_lacks_with_clean_tmdb_money": len(ex), "union_films": len(fu),
                     "mean_ln_ratio_usable": round(float(f["ln_gross_over_budget"].mean()), 3),
                     "mean_ln_ratio_wikipedia_lacks": round(float(ex["ln_gross_over_budget"].mean()), 3),
                     "share_below_budget_usable": round(float((f["ln_gross_over_budget"] < 0).mean()), 3),
                     "share_below_budget_wikipedia_lacks": round(float((ex["ln_gross_over_budget"] < 0).mean()), 3)}

    # C rating
    cert = t["certification_us"].reindex(f["imdb_id"]).to_numpy()
    rating = np.where(pd.Series(cert).isin(RATINGS), cert, "other")
    y = net_return(f)
    ctrl = pd.get_dummies(f["era_bin"], drop_first=True, dtype=float).to_numpy()
    Rd = pd.get_dummies(pd.Series(rating), dtype=float)
    Rd = Rd[[c for c in Rd.columns if c != "R"]]           # R is the reference
    coefs = []
    for label, X in (("genres only", M), ("genres + rating", np.hstack([M, Rd.to_numpy()]))):
        fit = sm.OLS(y, sm.add_constant(np.hstack([X, ctrl]))).fit(cov_type="HC3")
        b, ci = fit.params[1:1 + M.shape[1]], fit.conf_int()[1:1 + M.shape[1]]
        m = b.mean()
        coefs += [{"model": label, "term": g, "coef_centred": round(b[j] - m, 4), "lo": round(ci[j, 0] - m, 4), "hi": round(ci[j, 1] - m, 4)}
                  for j, g in enumerate(keep)]
        if label == "genres + rating":
            k = M.shape[1]
            coefs += [{"model": label, "term": f"rating {c} vs R", "coef_centred": round(fit.params[1 + k + i], 4),
                       "lo": round(fit.conf_int()[1 + k + i, 0], 4), "hi": round(fit.conf_int()[1 + k + i, 1], 4)}
                      for i, c in enumerate(Rd.columns)]
    pd.DataFrame(coefs).to_csv(OUT / "rating_genre_effects.csv", index=False)
    Wr = pd.get_dummies(pd.Series(rating), dtype=float)
    rrows = shuffle_test(f, f["ln_gross_over_budget"].to_numpy(), Wr.to_numpy(), a.perms, SEED + 3020, "rating group", list(Wr.columns))
    pd.DataFrame(rrows).to_csv(OUT / "rating_tests.csv", index=False)
    share = pd.DataFrame({g: pd.Series(rating[M[:, j] == 1]).value_counts(normalize=True) for j, g in enumerate(keep)}).T.fillna(0).round(3)
    share.index.name = "genre"
    share.to_csv(OUT / "rating_mix_by_genre.csv")
    piv = pd.DataFrame(coefs)
    piv = piv[piv["term"].isin(keep)].pivot(index="term", columns="model", values="coef_centred")
    summ["rating"] = {"films_with_rating": int(pd.Series(cert).isin(RATINGS).sum()),
                      "spearman_genre_effects_with_vs_without_rating": round(float(piv.corr(method="spearman").iloc[0, 1]), 3)}

    # D Horror subgenres from keywords
    hj = keep.index("Horror")
    hmask = M[:, hj] == 1
    fh = f[hmask].reset_index(drop=True)
    kw = t["keywords"].reindex(fh["imdb_id"]).fillna("").map(lambda s: {x.strip().lower() for x in s.split("|") if x.strip()})
    S = np.column_stack([kw.map(lambda k, s=s: bool(k & s)).to_numpy().astype(float) for s in SUBGENRES.values()])
    yh = net_return(f)[hmask]
    srows = shuffle_test(fh, yh, S, a.perms, SEED + 3030, "Horror subgenre (return net of budget and era)", list(SUBGENRES))
    for r_ in srows:
        col = S[:, list(SUBGENRES).index(r_["genre"])]
        r_["mean_other_horror"] = round(float(yh[col == 0].mean()), 4)
        r_["median"] = round(float(np.median(yh[col == 1])), 4)
        r_["median_other_horror"] = round(float(np.median(yh[col == 0])), 4)
        r_["share_beating_other_horror_median"] = round(float((yh[col == 1] > np.median(yh[col == 0])).mean()), 3)
    pd.DataFrame(srows).rename(columns={"genre": "subgenre"}).to_csv(OUT / "horror_subgenres.csv", index=False)
    summ["horror"] = {"films": int(hmask.sum()), "with_keywords": int((kw.map(len) > 0).sum()),
                      "in_any_subgenre": int((S.sum(1) > 0).sum())}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(pd.DataFrame(agree).to_string(index=False))
    d = pd.DataFrame(rows)
    print(d.pivot(index="genre", columns="run", values="excess").round(3).to_string())
    print(d.pivot(index="genre", columns="run", values="p_holm").round(3).to_string())
    print(piv.round(3).to_string())
    print(pd.DataFrame(coefs).query("term.str.startswith('rating')", engine="python").to_string(index=False))
    print(pd.DataFrame(rrows).to_string(index=False))
    print(pd.DataFrame(srows).to_string(index=False))


if __name__ == "__main__":
    main()
