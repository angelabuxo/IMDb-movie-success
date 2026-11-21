"""Q1 follow-up checks, added after the main results (q1_genre.py) and kept apart from them. Each one asks what a
different choice would change; none replaces the main test.

  1 smooth     no budget bins: ln(gross / budget) minus a smooth fit on ln(budget) (natural cubic spline, 5 df)
               and era, then genre means of that leftover against a plain shuffle of it
  2 tiers      genre x fixed budget tier (under $5M, $5-30M, $30-100M, $100M and up): the excess inside each tier,
               shuffled within budget deciles among the tier's films; cells with under MIN_CELL films are left blank
  3 trim       each genre's mean without its k highest ratios (k = 5, 10), against a shuffle that drops the top k
               the same way; the top films are listed by title and year
  3b outliers  the main test after removing the most extreme films at both ends across all films (the 10 highest and
               10 lowest ratios, then the top and bottom 1 % and 5 %), budget deciles recut; the 20 removed films listed
  4 matched    every film of a genre against the K_MATCH films outside it with the nearest ln(budget) in the same
               era bin; the mean difference with a 95 % bootstrap interval over the genre's films
  5 hit share  share of the genre's films in the top quarter of ln(gross / budget) (quartiles cut within era bin),
               and share with gross below budget, both against a shuffle within budget deciles

Same films, genres, seed and Holm correction as q1_genre.py, on `usable_money` only. Outputs in outputs/checks/.

Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.q1_checks [--perms 10000]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.genre.q1_genre import OUT as MAIN_OUT, SEED, holm, load, permuted_means, prepare, strata

OUT = MAIN_OUT / "checks"
TIERS = [(0, 5e6, "under $5M"), (5e6, 30e6, "$5-30M"), (30e6, 100e6, "$30-100M"), (100e6, np.inf, "$100M+")]
MIN_CELL = 20
TRIM = (5, 10)
OUTLIER_CUTS = (("10 highest and 10 lowest films", 10), ("top and bottom 1%", 0.01), ("top and bottom 5%", 0.05))
K_MATCH = 5
N_BOOT = 2000


def stats(obs, nl, perms):
    mu, sd = nl.mean(0), nl.std(0, ddof=1)
    p = (1 + (np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12).sum(0)) / (1 + perms)
    return mu, sd, p


def smooth(f, keep, M, perms):
    import statsmodels.formula.api as smf
    fit = smf.ols("y ~ cr(lb, df=5, constraints=\"center\") + C(era)", data=pd.DataFrame(
        {"y": f["ln_gross_over_budget"], "lb": f["ln_budget"], "era": f["era_bin"]})).fit()
    r = fit.resid.to_numpy()
    obs = r @ M / M.sum(0)
    nl = permuted_means(r, M, np.zeros(len(r), int), perms, np.random.default_rng(SEED + 101))
    mu, sd, p = stats(obs, nl, perms)
    ph = holm(p)
    return [{"check": "smooth", "genre": g, "n_films": int(M[:, j].sum()), "excess": round(obs[j] - mu[j], 4),
             "lo95": round(np.quantile(nl[:, j] - mu[j], .025), 4), "hi95": round(np.quantile(nl[:, j] - mu[j], .975), 4),
             "z": round((obs[j] - mu[j]) / sd[j], 2), "p_holm": round(ph[j], 5)} for j, g in enumerate(keep)]


def tiers(f, keep, M, perms):
    y = f["ln_gross_over_budget"].to_numpy()
    rows = []
    for t, (lo, hi, name) in enumerate(TIERS):
        idx = np.flatnonzero((f["budget"] >= lo) & (f["budget"] < hi))
        Mt, yt = M[idx], y[idx]
        n = Mt.sum(0)
        ok = n >= MIN_CELL
        obs = np.where(ok, yt @ Mt / np.maximum(n, 1), np.nan)
        nl = permuted_means(yt, Mt[:, ok], strata(f.iloc[idx], "budget"), perms, np.random.default_rng(SEED + 200 + t))
        mu, sd, p = stats(obs[ok], nl, perms)
        k = 0
        for j, g in enumerate(keep):
            row = {"check": "tiers", "tier": name, "genre": g, "n_films": int(n[j])}
            if ok[j]:
                row.update({"mean": round(obs[j], 4), "excess": round(obs[j] - mu[k], 4), "z": round((obs[j] - mu[k]) / sd[k], 2),
                            "p": p[k]})
                k += 1
            rows.append(row)
    d = pd.DataFrame(rows)
    has = d["p"].notna()
    d.loc[has, "p_holm"] = holm(d.loc[has, "p"].to_numpy()).round(5)
    d["p"] = d["p"].round(5)
    return d


def trimmed_means(y, M, k):
    """Genre means of y without each genre's k largest values."""
    out = np.empty(M.shape[1])
    for j in range(M.shape[1]):
        v = y[M[:, j] == 1]
        out[j] = np.sort(v)[:-k].mean()
    return out


def trim(f, keep, M, perms):
    y = f["ln_gross_over_budget"].to_numpy()
    groups = strata(f, "budget")
    idx_by_group = [np.flatnonzero(groups == s) for s in np.unique(groups)]
    cols = [np.flatnonzero(M[:, j]) for j in range(M.shape[1])]
    rng = np.random.default_rng(SEED + 300)
    rows, tops = [], []
    for k in TRIM:
        obs = trimmed_means(y, M, k)
        nl = np.empty((perms, len(keep)))
        yp = y.copy()
        for p in range(perms):
            for idx in idx_by_group:
                yp[idx] = y[rng.permutation(idx)]
            for j, c in enumerate(cols):
                v = yp[c]
                nl[p, j] = (v.sum() - np.partition(v, len(v) - k)[-k:].sum()) / (len(v) - k)
        mu, sd, pv = stats(obs, nl, perms)
        ph = holm(pv)
        rows += [{"check": f"trim_top{k}", "genre": g, "n_films": int(M[:, j].sum()) - k, "mean": round(obs[j], 4),
                  "excess": round(obs[j] - mu[j], 4), "z": round((obs[j] - mu[j]) / sd[j], 2), "p_holm": round(ph[j], 5)}
                 for j, g in enumerate(keep)]
    for j, g in enumerate(keep):
        sub = f.iloc[cols[j]].nlargest(max(TRIM), "ln_gross_over_budget")
        tops += [{"genre": g, "rank": r + 1, "imdb_id": s.imdb_id, "title": s.title, "year": int(s.year),
                  "budget": s.budget, "gross": s.gross, "ln_ratio": round(s.ln_gross_over_budget, 3)}
                 for r, s in enumerate(sub.itertuples())]
    return rows, tops


def outliers(f, keep, M, perms):
    """The main test after removing the most extreme ratios at both ends, across all films: the k highest and k
    lowest ln(gross / budget), or a share of films at each end. Budget deciles are recut on the remaining films."""
    y = f["ln_gross_over_budget"].to_numpy()
    r = pd.Series(y).rank(method="first").to_numpy()
    rows, removed = [], []
    for name, cut in OUTLIER_CUTS:
        k = int(cut) if cut >= 1 else int(round(cut * len(y)))
        keep_rows = (r > k) & (r <= len(y) - k)
        fk = f[keep_rows].reset_index(drop=True)
        fk["budget_bin"] = pd.qcut(np.log(fk["budget"]).rank(method="first"), 10, labels=False)
        Mk, yk = M[keep_rows], y[keep_rows]
        obs = yk @ Mk / Mk.sum(0)
        nl = permuted_means(yk, Mk, strata(fk, "budget"), perms, np.random.default_rng(SEED + 350 + len(rows)))
        mu, sd, p = stats(obs, nl, perms)
        ph = holm(p)
        rows += [{"check": name, "films_removed": int((~keep_rows).sum()), "genre": g, "n_films": int(Mk[:, j].sum()),
                  "excess": round(obs[j] - mu[j], 4), "z": round((obs[j] - mu[j]) / sd[j], 2), "p_holm": round(ph[j], 5)}
                 for j, g in enumerate(keep)]
        if cut == 10:
            gone = f[~keep_rows].assign(side=np.where(r[~keep_rows] > len(y) - k, "highest", "lowest"))
            removed = gone[["side", "imdb_id", "title", "year", "genres", "budget", "gross", "ln_gross_over_budget"]].sort_values(
                "ln_gross_over_budget", ascending=False)
    return rows, removed


def matched(f, keep, M):
    y = f["ln_gross_over_budget"].to_numpy()
    lb = f["ln_budget"].to_numpy()
    era = f["era_bin"].to_numpy()
    rng = np.random.default_rng(SEED + 400)
    rows = []
    for j, g in enumerate(keep):
        inside = M[:, j] == 1
        diffs = []
        for i in np.flatnonzero(inside):
            pool = np.flatnonzero(~inside & (era == era[i]))
            if len(pool) < K_MATCH:
                continue
            near = pool[np.argpartition(np.abs(lb[pool] - lb[i]), K_MATCH - 1)[:K_MATCH]]
            diffs.append(y[i] - y[near].mean())
        diffs = np.array(diffs)
        boot = np.array([rng.choice(diffs, len(diffs)).mean() for _ in range(N_BOOT)])
        rows.append({"check": "matched", "genre": g, "n_films": len(diffs), "excess": round(diffs.mean(), 4),
                     "lo95": round(np.quantile(boot, .025), 4), "hi95": round(np.quantile(boot, .975), 4),
                     "excludes_zero": bool(np.quantile(boot, .025) > 0 or np.quantile(boot, .975) < 0)})
    return rows


def shares(f, keep, M, perms):
    q75 = f.groupby("era_bin")["ln_gross_over_budget"].transform(lambda s: s.quantile(.75))
    rows = []
    for name, ind in (("hit_share", (f["ln_gross_over_budget"] >= q75)), ("below_budget_share", f["ln_gross_over_budget"] < 0)):
        y = ind.to_numpy().astype(float)
        obs = y @ M / M.sum(0)
        nl = permuted_means(y, M, strata(f, "budget"), perms, np.random.default_rng(SEED + 500 + len(rows)))
        mu, sd, p = stats(obs, nl, perms)
        ph = holm(p)
        rows += [{"check": name, "genre": g, "n_films": int(M[:, j].sum()), "share": round(obs[j], 4),
                  "expected": round(mu[j], 4), "excess_pp": round(100 * (obs[j] - mu[j]), 2),
                  "lo95_pp": round(100 * (np.quantile(nl[:, j], .025) - mu[j]), 2),
                  "hi95_pp": round(100 * (np.quantile(nl[:, j], .975) - mu[j]), 2), "p_holm": round(ph[j], 5)}
                 for j, g in enumerate(keep)]
    return rows


def heatmap(d, main, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = main.sort_values("excess")["genre"].tolist()[::-1]
    names = [t[2] for t in TIERS]
    E = d.pivot(index="genre", columns="tier", values="excess").reindex(index=order, columns=names)
    P = d.pivot(index="genre", columns="tier", values="p_holm").reindex(index=order, columns=names)
    N = d.pivot(index="genre", columns="tier", values="n_films").reindex(index=order, columns=names)
    lim = np.nanmax(np.abs(E.to_numpy()))
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    cmap = matplotlib.colormaps["RdBu"].copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(E.to_numpy()), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(order)):
        for k in range(len(names)):
            e, p, n = E.iat[i, k], P.iat[i, k], N.iat[i, k]
            txt = f"n={int(n)}" if np.isnan(e) else f"{e:+.2f}{' *' if p < .05 else ''}\nn={int(n)}"
            ax.text(k, i, txt, ha="center", va="center", fontsize=6, color="#222" if np.isnan(e) or abs(e) < .6 * lim else "white")
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_yticks(range(len(order)), order, fontsize=8)
    ax.set_title("Excess ln(gross / budget) by genre and budget tier\n(shuffled within budget deciles; * Holm p < .05 "
                 f"over all cells; grey: under {MIN_CELL} films)", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=.6, label="excess over the shuffle")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    films, genres = load()
    f, keep, M = prepare(films, genres, "usable_money")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(smooth(f, keep, M, a.perms)).to_csv(OUT / "smooth.csv", index=False)
    d = tiers(f, keep, M, a.perms)
    d.to_csv(OUT / "tiers.csv", index=False)
    tr, tops = trim(f, keep, M, min(a.perms, 2000))
    pd.DataFrame(tr).to_csv(OUT / "trim.csv", index=False)
    ol, gone = outliers(f, keep, M, a.perms)
    pd.DataFrame(ol).to_csv(OUT / "outliers.csv", index=False)
    gone.to_csv(OUT / "outliers_removed.csv", index=False)
    pd.DataFrame(tops).to_csv(OUT / "top_films.csv", index=False)
    pd.DataFrame(matched(f, keep, M)).to_csv(OUT / "matched.csv", index=False)
    pd.DataFrame(shares(f, keep, M, a.perms)).to_csv(OUT / "shares.csv", index=False)
    main_sh = pd.read_csv(MAIN_OUT / "genre_shuffle.csv")
    main_sh = main_sh[(main_sh["set"] == "usable_money") & (main_sh["outcome"] == "ln_ratio") & (main_sh["null"] == "budget")]
    heatmap(d, main_sh, OUT / "figures" / "q1_genre_tiers")
    print("done", flush=True)


if __name__ == "__main__":
    main()
