"""Profit attribution: how much of a film's return goes with each person and company on it.

Run from the repo root, after the movie graph is built (about ten minutes):

    python -m analysis.attribution.run

Sample: the clean movies in the graph (USD budget and worldwide gross, 1975 to 2025, theatrical). Outcome:
y = ln(gross / budget). Gross over budget isn't profit: the budget leaves out marketing and cinemas keep part of
the gross. `model.py` describes the model; this script runs it and writes the numbers the notebook and the
explorer page read:

- outputs/summary.json: sample, penalties, how well it predicts, each block's share, nulls, reliability, and
  what the profit measures say about each other;
- outputs/entities.csv: every person and company with an effect, with a bootstrap interval and their dollars;
- outputs/films.csv: every film's return split into blocks, in logs and in dollars;
- outputs/film_entities.csv.gz: the same split down to each person and company on each film.

Two versions of every film's split are saved. "Out of fold" fits the effects without the film (and without the
other films of its franchise), using films from every year: it answers "how do this director's other films
do?". "Forward" fits them on films from earlier years only: it answers "what could have been known before
release?", and is the one to use for prediction.

Checks built in: returns shuffled among films of the same year (the model should then find nothing); two random
halves of the films fitted apart (the effects should agree); a plain shrunk mean of each entity's residuals as a
second method; a bootstrap over films for every effect.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import model as M

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("IMDB_DATA_DIR", REPO / "data" / "raw"))
OUT = Path(__file__).resolve().parent / "outputs"
SEED = 2805
CAST_TOP = 3
GRID = [1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1000.0, np.inf]
FORWARD_FROM = 1990
SHARES = [0.4, 0.5, 0.6]
PLAYERS = ["franchise"] + M.ROLES          # the blocks whose shares are computed; budget, year, genre are the base
BASE = ["budget", "year", "genre"]


def franchise_groups(fr: pd.DataFrame, films) -> np.ndarray:
    """A group label per film: films joined by follows, followed-by or a shared series are one group, so a
    sequel is never used to explain its own original."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for r in fr.itertuples():
        if not isinstance(r.qid, str):
            continue
        for col in ("follows", "followed_by", "series"):
            v = getattr(r, col)
            if isinstance(v, str):
                for q in v.replace(";", "|").replace(",", "|").split("|"):
                    if q.strip():
                        parent[find(r.qid)] = find(q.strip())
    qid = fr.set_index("tconst").qid.to_dict()
    return np.array([find(qid[t]) if isinstance(qid.get(t), str) else t for t in films], dtype=object)


def load():
    F = DATA / "features"
    n = pd.read_csv(F / "graph" / "nodes.csv", dtype={"id": str}, low_memory=False)
    m = n[(n.type == "movie") & (n.clean == 1)].set_index("id").sort_index()
    m = m[(m.budget_usd > 0) & (m.gross_usd > 0)].copy()
    m["log_budget"], m["log_mult"] = np.log(m.budget_usd), np.log(m.gross_usd / m.budget_usd)
    fr = pd.read_csv(F / "franchise.csv", dtype={"tconst": str})
    m["franchise"] = fr.set_index("tconst").franchise.reindex(m.index).fillna(0).astype(int)
    e = pd.read_csv(F / "graph" / "edges.csv", dtype={"source": str, "target": str}, low_memory=False)
    e = e[e.target.isin(m.index) & ((e.role != "actor") | (e.ordering <= CAST_TOP))]
    credits = pd.DataFrame({"film": e.target, "entity": e.source, "role": e.role.replace({"actor": "cast"})})
    names = n.set_index("id").label
    return m, credits, names, franchise_groups(fr, m.index)


def measures(m: pd.DataFrame) -> dict:
    """What the candidate profit measures say, and how far they agree on which films did best."""
    out = {"films": int(len(m)), "median_multiple": float(m.multiple.median()),
           "share_gross_under_budget": float((m.gross_usd < m.budget_usd).mean()), "scenarios": {}}
    lm = np.log(m.gross_usd / m.budget_usd)
    top_mult = set(lm.nlargest(len(m) // 10).index)
    for s in SHARES:
        p = s * m.gross_usd - m.budget_usd
        gains = p[p > 0].sort_values(ascending=False)
        top = set(p.nlargest(len(m) // 10).index)
        out["scenarios"][f"{s:g}"] = {
            "share_of_gross_kept": s, "break_even_multiple": 1 / s,
            "share_of_films_above_break_even": float((p > 0).mean()),
            "median_usd": float(p.median()), "total_usd": float(p.sum()),
            "share_of_all_gains_from_top_decile_of_films": float(gains.head(len(m) // 10).sum() / gains.sum()),
            "spearman_with_multiple": float(spearmanr(p, lm).statistic),
            "top_decile_overlap_with_multiple": len(top & top_mult) / len(top)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--bootstraps", type=int, default=200)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--folds", type=int, default=10)
    a = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    m, credits, names, groups = load()
    y = m.log_mult.to_numpy()
    year = m.year.to_numpy(int)
    d = M.build_design(m, credits)
    n_films = np.asarray(d.X.sum(axis=0)).ravel()
    log(f"{len(m):,} films, {d.X.shape[1]:,} columns, {len(set(groups)):,} franchise groups")

    # penalties: tuned on 5 folds, everything after uses its own folds
    pen, path = M.tune(d, y, M.grouped_folds(groups, 5, SEED), M.ROLES, GRID, log=log)
    folds = M.grouped_folds(groups, a.folds, SEED + 1)

    # how well it predicts, and each block's part
    pred, icpt, C = M.out_of_fold(d, y, pen, folds)
    blocks = M.CONTROLS + M.ROLES
    contrib = M.block_contributions(d, C, blocks)
    resid = y - pred
    r2_full, r2_base = M.r2(y, pred), M.cv_r2(d, y, pen, folds, blocks=BASE)
    log(f"cross-validated R2: base {r2_base:.4f}, full {r2_full:.4f}")
    cov_share = {b: float(np.cov(contrib[b], y)[0, 1] / y.var(ddof=1)) for b in blocks}

    f5 = M.grouped_folds(groups, 5, SEED + 2)
    shap, coalitions = M.shapley_blocks(lambda s: M.cv_r2(d, y, pen, f5, blocks=BASE + sorted(s)), PLAYERS)
    log("Shapley shares of cross-validated R2 done")
    alone = {p: coalitions[frozenset([p])] - coalitions[frozenset()] for p in PLAYERS}
    last = {p: coalitions[frozenset(PLAYERS)] - coalitions[frozenset(PLAYERS) - {p}] for p in PLAYERS}

    # null: the same films, returns shuffled within release year. For the whole model and for each block alone.
    rng = np.random.default_rng(SEED)
    null_gain, null_alone = [], {p: [] for p in PLAYERS}
    for k in range(a.shuffles):
        ys = M.shuffle_within(y, year, rng)
        b0_ = M.cv_r2(d, ys, pen, f5, blocks=BASE)
        null_gain.append(M.cv_r2(d, ys, pen, f5) - b0_)
        for p in PLAYERS:
            null_alone[p].append(M.cv_r2(d, ys, pen, f5, blocks=BASE + [p]) - b0_)
    obs_gain = coalitions[frozenset(PLAYERS)] - coalitions[frozenset()]
    log(f"shuffle null done: gain over base {obs_gain:.4f} against at most {max(null_gain):.4f} shuffled")

    # the effects themselves, on every film, with a bootstrap over franchise groups
    a_full, beta = M.fit(d, y, pen)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    boot = np.empty((a.bootstraps, len(beta)))
    for k in range(a.bootstraps):
        rows = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        boot[k] = M.fit(d, y, pen, rows=rows)[1]
    lo, hi = np.percentile(boot, [5, 95], axis=0)
    same_sign = (np.sign(boot) == np.sign(beta)).mean(axis=0)
    log("bootstrap done")

    rel = M.split_half(d, y, pen, groups, M.ROLES, a.splits, SEED + 3)
    log("split halves done")

    # second method: the shrunk mean of each entity's residuals from the controls alone
    a0, b0 = M.fit(d, y, pen, blocks=M.CONTROLS)
    res0 = y - a0 - d.X @ b0
    role_col = np.isin(d.block, M.ROLES)
    naive = np.zeros(len(beta))
    sums = np.asarray(d.X.T @ res0).ravel()
    for r in M.ROLES:
        c = d.cols(r)
        naive[c] = sums[c] / (n_films[c] + pen[r]) if np.isfinite(pen[r]) else 0.0
    second = {r: {"spearman": float(spearmanr(beta[d.cols(r)], naive[d.cols(r)]).statistic) if beta[d.cols(r)].std() > 0 else None,
                  "sd_joint": float(beta[d.cols(r)].std()), "sd_one_at_a_time": float(naive[d.cols(r)].std())} for r in M.ROLES}

    # forward: effects from earlier years only
    pred_f, icpt_f, C_f = M.forward(d, y, pen, year, FORWARD_FROM)
    contrib_f = M.block_contributions(d, C_f, blocks)
    okf = np.isfinite(pred_f)
    pred_fb = np.full(len(y), np.nan)
    for yv in sorted(set(year[okf])):
        te, tr = np.flatnonzero(year == yv), np.flatnonzero(year < yv)
        ab, bb = M.fit(d, y, pen, rows=tr, blocks=BASE)
        pred_fb[te] = ab + d.X[te] @ bb
    dm = lambda v: v - pd.Series(v).groupby(year).transform("mean").to_numpy()
    fwd = {"from_year": FORWARD_FROM, "films": int(okf.sum()), "r2_full": M.r2(y[okf], pred_f[okf]),
           "r2_base": M.r2(y[okf], pred_fb[okf]),
           "spearman_within_year_full": float(spearmanr(dm(y)[okf], dm(np.nan_to_num(pred_f))[okf]).statistic),
           "spearman_within_year_base": float(spearmanr(dm(y)[okf], dm(np.nan_to_num(pred_fb))[okf]).statistic)}
    log(f"forward R2 from {FORWARD_FROM}: base {fwd['r2_base']:.4f}, full {fwd['r2_full']:.4f}")

    # dollars: every film's gross split between its controls, its people and companies, and what's left
    ctrl_mean = {b: contrib[b].mean() for b in M.CONTROLS}
    base_ln = icpt + sum(ctrl_mean.values())
    base_gross = m.budget_usd.to_numpy() * np.exp(base_ln)
    usd = {b: np.zeros(len(y)) for b in blocks + ["unexplained"]}
    Cr = C[:, role_col].tocsr()
    rcols = np.flatnonzero(role_col)
    fe = []
    for i in range(len(y)):
        s, e_ = Cr.indptr[i], Cr.indptr[i + 1]
        cols = rcols[Cr.indices[s:e_]]
        terms = np.concatenate([[contrib[b][i] - ctrl_mean[b] for b in M.CONTROLS], Cr.data[s:e_], [resid[i]]])
        phi = M.shapley_product(base_gross[i], terms)
        for j, b in enumerate(M.CONTROLS):
            usd[b][i] = phi[j]
        usd["unexplained"][i] = phi[-1]
        for c, t, p in zip(cols, Cr.data[s:e_], phi[len(M.CONTROLS):-1]):
            usd[d.block[c]][i] += p
            fe.append((d.films[i], d.block[c], d.key[c], t, p))
    check = np.abs(sum(usd.values()) - (m.gross_usd.to_numpy() - base_gross)).max()
    assert check < 1.0, f"dollar parts don't add up (off by {check})"
    fe = pd.DataFrame(fe, columns=["tconst", "role", "id", "effect", "usd"])
    fe["name"] = fe.id.map(names)
    log("dollar split done")

    # write
    OUT.mkdir(exist_ok=True)
    films = pd.DataFrame({"tconst": m.index, "title": m.label, "year": year, "genres": m.genres, "franchise": m.franchise,
                          "budget_usd": m.budget_usd, "gross_usd": m.gross_usd, "log_mult": y, "predicted": pred,
                          "unexplained": resid, "baseline_gross_usd": base_gross})
    for b in blocks:
        films[f"ln_{b}"] = contrib[b] - ctrl_mean.get(b, 0.0)
    for b in blocks + ["unexplained"]:
        films[f"usd_{b}"] = usd[b]
    films["predicted_forward"] = pred_f
    for b in blocks:
        films[f"fwd_{b}"] = np.where(okf, contrib_f[b], np.nan)
    films.round(6).to_csv(OUT / "films.csv", index=False)
    fe[["tconst", "role", "id", "name", "effect", "usd"]].round(6).to_csv(OUT / "film_entities.csv.gz", index=False)

    tot = fe.groupby(["role", "id"]).usd.agg(["sum", "mean"])
    span = credits.assign(year=credits.film.map(m.year)).groupby(["role", "entity"]).year.agg(["min", "max"])
    ent = pd.DataFrame({"role": d.block[role_col], "id": d.key[role_col], "films": n_films[role_col].astype(int),
                        "effect": beta[role_col], "lo90": lo[role_col], "hi90": hi[role_col],
                        "same_sign_share": same_sign[role_col], "factor": np.exp(beta[role_col]),
                        "one_at_a_time": naive[role_col]})
    ent["name"] = ent.id.map(names)
    ent = ent.join(tot.rename(columns={"sum": "usd_total", "mean": "usd_per_film"}), on=["role", "id"])
    ent = ent.join(span.rename(columns={"min": "first_year", "max": "last_year"}), on=["role", "id"])
    ent = ent[["role", "id", "name", "films", "first_year", "last_year", "effect", "lo90", "hi90", "same_sign_share",
               "factor", "one_at_a_time", "usd_total", "usd_per_film"]]
    ent.sort_values(["role", "effect"], ascending=[True, False]).round(6).to_csv(OUT / "entities.csv", index=False)

    summary = {
        "sample": {"films": int(len(m)), "years": [int(year.min()), int(year.max())], "franchise_films": int(m.franchise.sum()),
                   "franchise_groups": int(len(uniq)), "cast_top": CAST_TOP, "min_films_for_an_effect": 2,
                   "entities_with_an_effect": {r: int(len(d.cols(r))) for r in M.ROLES},
                   "films_with_at_least_one": {r: int((np.asarray(d.X[:, d.cols(r)].sum(axis=1)).ravel() > 0).sum()) for r in M.ROLES}},
        "outcome": {"mean": float(y.mean()), "sd": float(y.std(ddof=1))},
        "penalties": {r: (None if not np.isfinite(pen[r]) else pen[r]) for r in M.ROLES},
        "effect_sd": {r: float(beta[d.cols(r)].std()) for r in M.ROLES},
        "cv": {"folds": a.folds, "r2_base": r2_base, "r2_full": r2_full,
               "covariance_share": cov_share, "covariance_share_unexplained": float(np.cov(resid, y)[0, 1] / y.var(ddof=1))},
        "shapley": {"folds": 5, "base_blocks": BASE, "r2_base": coalitions[frozenset()], "r2_all": coalitions[frozenset(PLAYERS)],
                    "share": shap, "alone_on_base": alone, "lost_when_dropped": last},
        "null": {"shuffles": a.shuffles, "what": "returns shuffled among films of the same release year",
                 "gain_over_base_observed": obs_gain, "gain_over_base_null_mean": float(np.mean(null_gain)),
                 "gain_over_base_null_max": float(np.max(null_gain)),
                 "gain_over_base_null_p95": float(np.percentile(null_gain, 95)),
                 "alone_on_base_null_mean": {p: float(np.mean(v)) for p, v in null_alone.items()},
                 "alone_on_base_null_max": {p: float(np.max(v)) for p, v in null_alone.items()},
                 "alone_on_base_shuffles_at_or_above_observed": {p: int((np.array(v) >= alone[p]).sum()) for p, v in null_alone.items()}},
        "split_half": {"splits": a.splits, "min_films_each_half": 3, "by_role": rel},
        "second_method": {"what": "shrunk mean of each entity's residuals from the controls, one entity at a time", "by_role": second},
        "bootstrap": {"resamples": a.bootstraps, "unit": "franchise group",
                      "share_of_effects_with_interval_off_zero": {r: float(((lo[d.cols(r)] > 0) | (hi[d.cols(r)] < 0)).mean()) for r in M.ROLES}},
        "forward": fwd,
        "dollars": {"what": "Shapley split of gross minus baseline gross (budget times the average multiple), nominal USD",
                    "net_by_block": {b: float(v.sum()) for b, v in usd.items()},
                    "absolute_by_block": {b: float(np.abs(v).sum()) for b, v in usd.items()},
                    "total_gross": float(m.gross_usd.sum()), "total_baseline_gross": float(base_gross.sum())},
        "measures": measures(m),
        "tuning_path": [{"role": r, "penalty": (None if not np.isfinite(g) else g), "cv_r2": s} for r, g, s in path],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"wrote {OUT}")


if __name__ == "__main__":
    main()
