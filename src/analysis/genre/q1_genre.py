"""Q1. Do some genres earn more than others, once we compare films with similar budgets?

Settings, fixed before any genre result was looked at:
  Films     `usable_money` in features/film_analysis_set.csv (the shared primary rule, which calls
            analysis/clean_rule.py). Head-to-head check on `in_common_set` (the films Q2 and Q3 also use).
  Outcome   ln(gross / budget), the `ln_gross_over_budget` column. Gross over budget isn't profit: marketing isn't
            in the budget and cinemas keep part of the gross.
  Genres    the IMDb genres in features/film_genres.csv. A film counts in every genre it has, so genres overlap
            and their tests aren't independent. Genres with fewer than MIN_FILMS films are left out.
  Statistic the mean ln(gross / budget) of the films tagged with the genre.
  Nulls     the returns are shuffled among the films, the genre tags stay put, and the genre means are recomputed.
              none            shuffle across all films (what "this genre earns more" means without any matching)
              budget          shuffle only within deciles of ln(budget): the primary null, so a genre can't win
                              just by being cheap
              budget_era      shuffle within ln(budget) decile x era bin, because returns fall strongly by decade
              budget_fine     shuffle within 50 equal-sized ln(budget) bins (added after seeing that the lowest decile
                              spans $5k to $3.2M, where most cheap horror films sit)
            Excess = observed mean - null mean, z = excess / null sd, two-sided p with the +1 correction, and a
            Holm correction over the genres of one null.
  Outliers  a handful of micro-budget hits (and a few near-total flops) sit far out on the log scale, so the
            budget null is rerun on the percentile rank of ln(gross / budget), where each film counts once.
  Second    OLS of ln(gross / budget) on every kept genre (multi-hot), ln(budget) decile and era dummies, with HC3
  method    standard errors. Its genre coefficients are partial effects (holding the film's other genres fixed),
            so they answer a slightly different question than the shuffle. Almost every film has a kept genre, so the
            raw coefficients sit against a near-empty reference; the check is the rank correlation of the centred
            coefficients with the budget x era shuffle excess. The same fits give the R2 that genre adds to budget + era.
  Not used  the vote tier: votes are partly a consequence of success, so matching on them would match on the outcome.

Outputs (numbers only, no text from any film) in analysis/genre/outputs/:
  genre_shuffle.csv     one row per set x null x genre
  genre_ols.csv         one row per set x genre
  overlap.csv           films shared by each pair of kept genres (usable_money set)
  summary.json          the settings and the headline counts
  figures/q1_genre.png and .pdf   bars of the observed mean with the budget-matched shuffle band

Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_genre [--perms 10000]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("IMDB_DATA_DIR", ROOT / "data" / "raw"))
OUT = Path(__file__).resolve().parent / "outputs"
SEED = 2805
MIN_FILMS = 100
N_BUDGET_BINS = 10
NULLS = ("none", "budget", "budget_era", "budget_fine")
N_FINE_BINS = 50
SETS = ("usable_money", "in_common_set")


def load():
    films = pd.read_csv(DATA / "features" / "film_analysis_set.csv")
    genres = pd.read_csv(DATA / "features" / "film_genres.csv")
    return films, genres


def prepare(films, genres, flag, keep=None):
    """Films of one set with outcome, strata and a film x genre membership matrix."""
    f = films[films[flag] & films["ln_gross_over_budget"].notna()].copy().reset_index(drop=True)
    f["ln_budget"] = np.log(f["budget"])
    # deciles of ln(budget) within the set (ties broken by rank, so the bins are equal-sized)
    f["budget_bin"] = pd.qcut(f["ln_budget"].rank(method="first"), N_BUDGET_BINS, labels=False)
    f["budget_fine"] = pd.qcut(f["ln_budget"].rank(method="first"), N_FINE_BINS, labels=False)
    g = genres[genres["imdb_id"].isin(f["imdb_id"])]
    if keep is None:
        counts = g["genre"].value_counts()
        keep = sorted(counts[counts >= MIN_FILMS].index)
    pos = {i: k for k, i in enumerate(f["imdb_id"])}
    M = np.zeros((len(f), len(keep)))
    col = {gn: j for j, gn in enumerate(keep)}
    for i, gn in zip(g["imdb_id"], g["genre"]):
        if gn in col:
            M[pos[i], col[gn]] = 1.0
    return f, keep, M


def strata(f, null):
    if null == "none":
        return np.zeros(len(f), int)
    if null == "budget":
        return f["budget_bin"].to_numpy()
    if null == "budget_fine":
        return f["budget_fine"].to_numpy()
    return pd.factorize(f["budget_bin"].astype(str) + "|" + f["era_bin"].astype(str))[0]


def permuted_means(y, M, groups, perms, rng):
    """Genre means of y after shuffling y within each group, perms times. Returns (perms, n_genres)."""
    n_per = M.sum(0)
    idx_by_group = [np.flatnonzero(groups == s) for s in np.unique(groups)]
    out = np.empty((perms, M.shape[1]))
    yp = y.copy()
    for p in range(perms):
        for idx in idx_by_group:
            yp[idx] = y[rng.permutation(idx)]
        out[p] = yp @ M / n_per
    return out


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def shuffle_table(f, keep, M, set_name, perms, outcome="ln_ratio", nulls=NULLS):
    y = f["ln_gross_over_budget"].to_numpy()
    if outcome == "rank":
        y = f["ln_gross_over_budget"].rank(pct=True).to_numpy()
    obs = y @ M / M.sum(0)
    rows = []
    for k, null in enumerate(nulls):
        rng = np.random.default_rng(SEED + k + (10 if outcome == "rank" else 0))
        nl = permuted_means(y, M, strata(f, null), perms, rng)
        mu, sd = nl.mean(0), nl.std(0, ddof=1)
        dev = np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12
        p = (1 + dev.sum(0)) / (1 + perms)
        ph = holm(p)
        for j, gn in enumerate(keep):
            rows.append({"set": set_name, "outcome": outcome, "null": null, "genre": gn, "n_films": int(M[:, j].sum()),
                         "mean": round(obs[j], 4), "median": round(float(np.median(y[M[:, j] == 1])), 4),
                         "null_mean": round(mu[j], 4), "null_lo95": round(np.quantile(nl[:, j], .025), 4),
                         "null_hi95": round(np.quantile(nl[:, j], .975), 4), "excess": round(obs[j] - mu[j], 4),
                         "z": round((obs[j] - mu[j]) / sd[j], 2), "p": round(p[j], 5), "p_holm": round(ph[j], 5),
                         "perms": perms})
    return rows


def ols_table(f, keep, M, set_name):
    """Genre coefficients next to ln(budget) decile and era dummies. Almost every film has a kept genre, so the
    coefficients are relative to a near-empty reference; the centred coefficient (minus the mean over genres) is the
    one to compare across genres (its interval is the raw one shifted by the same mean, so it ignores the
    uncertainty of that mean). Also returns the R2 of budget + era with and without the genres."""
    import statsmodels.api as sm
    controls = pd.concat([pd.get_dummies(f["budget_bin"], prefix="bud", drop_first=True, dtype=float),
                          pd.get_dummies(f["era_bin"], prefix="era", drop_first=True, dtype=float)], axis=1)
    y = f["ln_gross_over_budget"].to_numpy()
    base = sm.OLS(y, sm.add_constant(controls)).fit()
    fit = sm.OLS(y, sm.add_constant(pd.concat([pd.DataFrame(M, columns=keep), controls], axis=1))).fit(cov_type="HC3")
    ci = fit.conf_int()
    mean_coef = float(np.mean([fit.params[g] for g in keep]))
    rows = [{"set": set_name, "genre": gn, "coef": round(fit.params[gn], 4), "coef_centred": round(fit.params[gn] - mean_coef, 4),
             "se_hc3": round(fit.bse[gn], 4), "ci95_lo": round(ci.loc[gn, 0], 4), "ci95_hi": round(ci.loc[gn, 1], 4),
             "ci95_lo_centred": round(ci.loc[gn, 0] - mean_coef, 4), "ci95_hi_centred": round(ci.loc[gn, 1] - mean_coef, 4),
             "p": round(fit.pvalues[gn], 5), "n": int(fit.nobs)} for gn in keep]
    r2 = {"r2_budget_era": round(base.rsquared, 4), "r2_budget_era_genre": round(fit.rsquared, 4),
          "r2_added_by_genre": round(fit.rsquared - base.rsquared, 4),
          "adj_r2_added_by_genre": round(fit.rsquared_adj - base.rsquared_adj, 4)}
    return rows, r2


def overlap(f, keep, M):
    C = M.T @ M
    return [{"genre_a": a, "genre_b": b, "films_both": int(C[i, j]),
             "share_of_a": round(C[i, j] / C[i, i], 3)}
            for i, a in enumerate(keep) for j, b in enumerate(keep) if i != j]


def figure(sh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    main = sh[(sh["set"] == "usable_money") & (sh["outcome"] == "ln_ratio")]
    d = main[main["null"] == "budget"].sort_values("mean")
    overall = float(main[main["null"] == "none"]["null_mean"].mean())
    colour = np.where(d["p_holm"] < .05, np.where(d["excess"] > 0, "#2f6db5", "#c0504d"), "#b8b8b8")
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    yy = np.arange(len(d))
    ax.barh(yy, d["mean"], color=colour, height=.55, zorder=2)
    for k, (_, r) in enumerate(d.iterrows()):
        ax.add_patch(plt.Rectangle((r["null_lo95"], k - .38), r["null_hi95"] - r["null_lo95"], .76, facecolor="#000",
                                   alpha=.12, edgecolor="#222", lw=.7, zorder=3))
    ax.axvline(overall, color="#555", lw=.8, ls=":", zorder=4)
    ax.axvline(0, color="#333", lw=.6)
    ax.set_yticks(yy, [f"{g} ({n})" for g, n in zip(d["genre"], d["n_films"])], fontsize=8)
    ax.set_ylim(-.7, len(d) - .3)
    ax.set_xlabel("mean ln(gross / budget)   (0 = gross equals budget, 0.69 = twice the budget)", fontsize=8)
    ax.set_title(f"Genre returns against returns shuffled within budget deciles ({len(d)} genres, "
                 f"{int(main['perms'].iloc[0]):,} shuffles)", fontsize=9)
    ax.legend(handles=[Patch(color="#2f6db5", label="above the band (Holm p < .05)"),
                       Patch(color="#c0504d", label="below the band (Holm p < .05)"),
                       Patch(color="#b8b8b8", label="inside the band"),
                       Patch(facecolor="#000", alpha=.12, edgecolor="#222", label="95 % of shuffles: films of the same budgets"),
                       Line2D([], [], color="#555", ls=":", label="all-film average")],
              fontsize=7, loc="upper center", bbox_to_anchor=(.45, -.1), ncol=3, frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    films, genres = load()
    OUT.mkdir(parents=True, exist_ok=True)
    sh, ols = [], []
    summary = {"min_films": MIN_FILMS, "budget_bins": N_BUDGET_BINS, "fine_budget_bins": N_FINE_BINS, "seed": SEED, "perms": a.perms, "sets": {}}
    for s in SETS:
        f, keep, M = prepare(films, genres, s)
        if s == "usable_money":
            pd.DataFrame(overlap(f, keep, M)).to_csv(OUT / "overlap.csv", index=False)
        sh += shuffle_table(f, keep, M, s, a.perms)
        sh += shuffle_table(f, keep, M, s, a.perms, outcome="rank", nulls=("budget",))
        rows, r2 = ols_table(f, keep, M, s)
        ols += rows
        summary["sets"][s] = {"films": len(f), "genres_kept": keep, "films_with_no_kept_genre": int((M.sum(1) == 0).sum()),
                              "mean_genres_per_film": round(float(M.sum(1).mean()), 2),
                              "all_film_mean_ln_ratio": round(float(f["ln_gross_over_budget"].mean()), 4), **r2}
        print(f"{s}: {len(f)} films, {len(keep)} genres", flush=True)
    sh, ols = pd.DataFrame(sh), pd.DataFrame(ols)
    from scipy.stats import spearmanr
    for s in SETS:
        a_ = sh[(sh["set"] == s) & (sh["outcome"] == "ln_ratio") & (sh["null"] == "budget_era")].set_index("genre")["excess"]
        b_ = ols[ols["set"] == s].set_index("genre")["coef_centred"].reindex(a_.index)
        summary["sets"][s]["spearman_ols_vs_shuffle_excess"] = round(float(spearmanr(a_, b_)[0]), 3)
    sh.to_csv(OUT / "genre_shuffle.csv", index=False)
    ols.to_csv(OUT / "genre_ols.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    figure(sh, OUT / "figures" / "q1_genre")
    sh["test"] = sh["outcome"] + ":" + sh["null"]
    show = sh[sh["set"] == "usable_money"].pivot(index="genre", columns="test", values=["excess", "p_holm"])
    print(show.round(3).sort_values(("excess", "ln_ratio:budget")).to_string())


if __name__ == "__main__":
    main()
