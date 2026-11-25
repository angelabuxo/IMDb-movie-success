"""Do the genre results survive giving every film without money a bad return? Declared in CHECKS.md before running.

Captures how far the ranking could move if every unpriced film had done badly. Misses the true returns of those films:
with one fill value, a genre's shift comes only from its share of unpriced films.

Outputs in outputs/worst_case/: tests.csv (one row per version x genre), verdict.csv, summary.json. Numbers only.
Run from src/:  IMDB_DATA_DIR=... python -m analysis.genre.q1_worst_case [--perms 10000]
"""
import argparse
import json

import numpy as np
import pandas as pd

from analysis.clean_rule import CURRENCY, MAX_YEAR_EXCLUSIVE, MIN_BUDGET_USD, MIN_YEAR_INCLUSIVE
from analysis.genre.q1_genre import N_BUDGET_BINS, OUT, SEED, holm, load, permuted_means, prepare

DEST = OUT / "worst_case"


def members(f, genres, keep):
    pos, col = {i: k for k, i in enumerate(f["imdb_id"])}, {g: j for j, g in enumerate(keep)}
    M = np.zeros((len(f), len(keep)))
    for i, g in zip(genres["imdb_id"], genres["genre"]):
        if i in pos and g in col:
            M[pos[i], col[g]] = 1
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    a = ap.parse_args()
    films, genres = load()
    priced, keep, _ = prepare(films, genres, "usable_money")
    v = float(np.sort(priced["ln_gross_over_budget"].to_numpy())[:len(priced) // 4].mean())
    f = films[(films["year"] >= MIN_YEAR_INCLUSIVE) & (films["year"] < MAX_YEAR_EXCLUSIVE)].reset_index(drop=True)
    f["y"] = f["ln_gross_over_budget"].where(f["usable_money"], v)
    none = f["money_exclude_reason"].eq("no budget or gross parsed")
    has_budget = (~f["usable_money"] & f["budget"].ge(MIN_BUDGET_USD) & f["budget_cur"].eq(CURRENCY) & f["budget_flag"].isna()
                  & f["gross"].isna())
    versions = {"priced only, era null": (f["usable_money"], "era"),
                "all unpriced filled, era null": (pd.Series(True, f.index), "era"),
                "unparsed only filled, era null": (f["usable_money"] | none, "era"),
                "priced only, budget null": (f["usable_money"], "budget"),
                "budget known but no gross filled, budget null": (f["usable_money"] | has_budget, "budget")}
    rows, count = [], {}
    for k, (name, (mask, null)) in enumerate(versions.items()):
        d = f[mask].reset_index(drop=True)
        M, y = members(d, genres, keep), d["y"].to_numpy()
        grp = (pd.factorize(d["era_bin"])[0] if null == "era" else
               pd.qcut(np.log(d["budget"]).rank(method="first"), N_BUDGET_BINS, labels=False).to_numpy())
        nl = permuted_means(y, M, grp, a.perms, np.random.default_rng(SEED + 400 + k))
        obs, mu, sd = y @ M / M.sum(0), nl.mean(0), nl.std(0, ddof=1)
        p = (1 + (np.abs(nl - mu) >= np.abs(obs - mu) - 1e-12).sum(0)) / (1 + a.perms)
        count[name] = {"films": len(d), "filled": int((~d["usable_money"]).sum())}
        for j, (g, h) in enumerate(zip(keep, holm(p))):
            un = 1 - (d["usable_money"].to_numpy() @ M[:, j]) / M[:, j].sum()
            rows.append({"version": name, "genre": g, "films": int(M[:, j].sum()), "share_unpriced": round(float(un), 4),
                         "mean": round(obs[j], 4), "excess": round(obs[j] - mu[j], 4), "z": round((obs[j] - mu[j]) / sd[j], 2),
                         "p_holm": round(h, 5)})
    t = pd.DataFrame(rows)
    # tipping point: the exact era-matched excess is linear in the fill value
    M = members(f, genres, keep)

    def excess(val):
        y = f["ln_gross_over_budget"].where(f["usable_money"], val)
        return (y.to_numpy() - y.groupby(f["era_bin"]).transform("mean").to_numpy()) @ M / M.sum(0)
    e0, e1 = excess(0.0), excess(1.0)
    tip = dict(zip(keep, np.where(np.abs(e1 - e0) > 1e-12, -e0 / (e1 - e0), np.nan)))
    W = t.pivot(index="genre", columns="version", values=["excess", "p_holm"])
    main_ = pd.read_csv(OUT / "genre_shuffle.csv").query("set == 'usable_money' and outcome == 'ln_ratio' and null == 'budget' and p_holm < .05")
    ver = []
    for _, r in main_.sort_values("excess", ascending=False).iterrows():
        x = W.loc[r.genre]
        e, ph = x[("excess", "all unpriced filled, era null")], x[("p_holm", "all unpriced filled, era null")]
        same = np.sign(e) == np.sign(r.excess)
        ver.append({"genre": r.genre, "main_excess": r.excess, "priced_era_excess": x[("excess", "priced only, era null")],
                    "filled_excess": e, "filled_p_holm": ph, "verdict": "holds" if same and ph < .05 else "weakens" if same else "flips",
                    "share_unpriced": t[(t.version == "all unpriced filled, era null") & (t.genre == r.genre)].share_unpriced.iloc[0],
                    "unparsed_excess": x[("excess", "unparsed only filled, era null")],
                    "unparsed_p_holm": x[("p_holm", "unparsed only filled, era null")],
                    "budget_filled_excess": x[("excess", "budget known but no gross filled, budget null")],
                    "budget_filled_p_holm": x[("p_holm", "budget known but no gross filled, budget null")],
                    "tipping_fill_value": round(float(tip[r.genre]), 3)})
    DEST.mkdir(parents=True, exist_ok=True)
    t.to_csv(DEST / "tests.csv", index=False)
    pd.DataFrame(ver).to_csv(DEST / "verdict.csv", index=False)
    (DEST / "summary.json").write_text(json.dumps({"fill_value": round(v, 4), "perms": a.perms, "films_1975_2025": len(f),
                                                   "unpriced": int((~f["usable_money"]).sum()), "versions": count}, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(json.dumps(count), v)
        print(pd.DataFrame(ver).to_string(index=False))
        print(W["excess"].round(3).to_string())


if __name__ == "__main__":
    main()
