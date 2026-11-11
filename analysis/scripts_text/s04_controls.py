"""Step 4: control for genre, era and budget by matching unprofitable scripts to profitable ones."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import s01_groups
from .config import cfg, log_run
from .paths import OUTPUTS
from .report import HIGH, HIGH_SYMBOL, LOW, LOW_SYMBOL, Fig, Md, Reading, Run, Table, style

MATCHES = OUTPUTS / "s04_matches.csv"
BALANCE = OUTPUTS / "s04_balance.csv"
SUMMARY = OUTPUTS / "s04_summary.json"


def frame() -> pd.DataFrame:
    """Scripted films with what matching needs; IMDb votes come from the private table."""
    r = s01_groups.load()
    sc = r["scripts"].copy()
    if r["clean"] is not None:
        sc = sc.merge(r["clean"][["tconst", "imdb_votes"]], on="tconst", how="left")
    sc["log_votes"] = np.log(sc.imdb_votes)
    # Scripts whose dialogue markup failed are never analysed, so they are not matched either.
    from .textstats import excluded_scripts
    return sc[~sc.tconst.isin(excluded_scripts())].reset_index(drop=True)


def match(sc: pd.DataFrame, caliper_sd: float, order_seed: int | None = None) -> pd.DataFrame:
    """Nearest-neighbour match on log budget, exact on primary genre and decade, without replacement.
    Treated = unprofitable under (A). By default the treated are matched hardest first (fewest
    candidates), ties by IMDb ID; order_seed shuffles the order instead (a stability check)."""
    cal = caliper_sd * sc.log_budget.std()
    treated = sc[sc.group_A == "low"]
    pool = sc[sc.group_A == "high"]
    key = lambda d: d.primary_genre + "|" + d.decade_bin
    tk, pk = key(treated), key(pool)
    n_cand = {t: int(((pk == k) & ((pool.log_budget - b).abs() <= cal)).sum())
              for t, k, b in zip(treated.tconst, tk, treated.log_budget)}
    if order_seed is None:
        order = sorted(treated.tconst, key=lambda t: (n_cand[t], t))
    else:
        order = list(np.random.default_rng(order_seed).permutation(sorted(treated.tconst)))
    used, rows = set(), []
    tt = treated.set_index("tconst")
    for t in order:
        k, b = tk[treated.tconst == t].iloc[0], tt.loc[t, "log_budget"]
        cand = pool[(pk == k) & ~pool.tconst.isin(used)]
        cand = cand.assign(dist=(cand.log_budget - b).abs())
        cand = cand[cand.dist <= cal].sort_values(["dist", "tconst"])
        if len(cand):
            c = cand.iloc[0]
            used.add(c.tconst)
            rows.append({"treated": t, "control": c.tconst, "cell": k, "budget_gap_log": float(c.dist)})
        else:
            rows.append({"treated": t, "control": None, "cell": k, "budget_gap_log": None})
    return pd.DataFrame(rows)


def matched_labels(sc: pd.DataFrame, m: pd.DataFrame) -> pd.Series:
    ok = m.dropna(subset=["control"])
    lab = pd.Series(None, index=sc.tconst, dtype=object)
    lab[ok.treated] = "low"
    lab[ok.control] = "high"
    return lab


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference: (mean_a - mean_b) / pooled SD. |SMD| < 0.1 is the usual
    'balanced' rule of thumb."""
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def balance(sc: pd.DataFrame, lab: pd.Series) -> pd.DataFrame:
    before_lo, before_hi = sc[sc.group_A == "low"], sc[sc.group_A == "high"]
    s = sc.set_index("tconst")
    after_lo, after_hi = s.loc[lab[lab == "low"].index], s.loc[lab[lab == "high"].index]
    rows = []
    for var, label in (("year", "release year"), ("log_budget", "log budget"), ("log_votes", "log IMDb votes")):
        rows.append({"variable": label, "SMD before": smd(before_lo[var], before_hi[var]),
                     "SMD after": smd(after_lo[var], after_hi[var])})
    for g in cfg()["groups"]["primary_genre_priority"]:
        rows.append({"variable": f"genre: {g}",
                     "SMD before": smd((before_lo.primary_genre == g).astype(float), (before_hi.primary_genre == g).astype(float)),
                     "SMD after": smd((after_lo.primary_genre == g).astype(float), (after_hi.primary_genre == g).astype(float))})
    t = pd.DataFrame(rows)
    t[["SMD before", "SMD after"]] = t[["SMD before", "SMD after"]].round(3)
    return t


def run() -> dict:
    sc = frame()
    m = match(sc, cfg()["matching"]["caliper_sd_log_budget"])
    m.to_csv(MATCHES, index=False)
    lab = matched_labels(sc, m)
    bal = balance(sc, lab)
    bal.to_csv(BALANCE, index=False)
    summary = {"treated": int(len(m)), "matched": int(m.control.notna().sum()),
               "unmatched_dropped": int(m.control.isna().sum()),
               "caliper_sd": cfg()["matching"]["caliper_sd_log_budget"],
               "caliper_log_units": float(cfg()["matching"]["caliper_sd_log_budget"] * sc.log_budget.std()),
               "median_budget_gap_log": float(m.budget_gap_log.median()),
               "max_abs_smd_before": float(bal[bal.variable != "log IMDb votes"]["SMD before"].abs().max()),
               "max_abs_smd_after": float(bal[bal.variable != "log IMDb votes"]["SMD after"].abs().max()),
               "smd_votes_before": float(bal.set_index("variable").loc["log IMDb votes", "SMD before"]),
               "smd_votes_after": float(bal.set_index("variable").loc["log IMDb votes", "SMD after"])}
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s04_controls", [MATCHES.name, BALANCE.name, SUMMARY.name], {"matching": cfg()["matching"]})
    return load()


def load() -> dict:
    return {"matches": pd.read_csv(MATCHES, dtype={"treated": str, "control": str}),
            "balance": pd.read_csv(BALANCE), "summary": json.loads(SUMMARY.read_text())}


def labels() -> pd.Series:
    """The matched sample as a label series indexed by tconst (for Step 5 onwards)."""
    sc = s01_groups.load()["scripts"]
    return matched_labels(sc, load()["matches"])


def tables(res: dict) -> dict:
    return {"balance": res["balance"]}


def figures(res: dict) -> dict:
    b = res["balance"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b["SMD before"], y=b.variable, mode="markers", name="before matching",
                             marker=dict(color=LOW, symbol=LOW_SYMBOL, size=11)))
    fig.add_trace(go.Scatter(x=b["SMD after"], y=b.variable, mode="markers", name="after matching",
                             marker=dict(color=HIGH, symbol=HIGH_SYMBOL, size=11)))
    for x in (-0.1, 0.1):
        fig.add_vline(x=x, line_dash="dot", line_color="#666")
    fig.add_vline(x=0, line_color="#999")
    fig.update_layout(xaxis_title="standardised mean difference, unprofitable − profitable (dotted: ±0.1)",
                      yaxis_autorange="reversed")
    return {"balance": style(fig, "Balance before and after matching", height=440)}


def reading(res: dict) -> str:
    s = res["summary"]
    return (f"Of the {s['treated']} unprofitable scripts, {s['matched']} find a profitable partner in the same "
            f"genre and decade with a log budget within {s['caliper_log_units']:.2f} (that is "
            f"{s['caliper_sd']} standard deviations); {s['unmatched_dropped']} have none and are dropped from the "
            f"matched comparison. The median budget gap within a pair is {s['median_budget_gap_log']:.2f} in log "
            f"units (about {np.expm1(s['median_budget_gap_log']):.0%}). On the variables we match on (year, budget, "
            f"genre) the largest absolute standardised mean difference falls from {s['max_abs_smd_before']:.2f} to "
            f"{s['max_abs_smd_after']:.2f}. IMDb votes are deliberately not matched on, because votes come after "
            f"release and partly measure success itself; their SMD stays at {s['smd_votes_after']:.2f} "
            f"(before: {s['smd_votes_before']:.2f}): unprofitable films are far less watched, so any word difference "
            f"in the matched sample may also reflect popularity.")


SECTION = [
    Md("""## Step 4 · Controlling for genre, era and budget

**What we do.** Without controls, a comparison of successful and unsuccessful scripts mostly shows genre and
era words: action against drama, 1990s against 2010s. We control in two ways.

1. **Stratification.** Definition (C) already compares films only within the same genre × decade cell, and
   Step 5 also runs within single genres and decades where both groups have at least 30 scripts.
2. **Matching.** For each unprofitable script (gross below budget) we find the profitable script with the same
   primary genre, the same decade and the closest log budget, within a caliper of 0.5 standard deviations of
   log budget, without reusing a partner. Scripts with no partner inside the caliper are dropped and counted.

**Why match instead of randomly undersampling.** Scattertext compares *rates* per 10,000 words, so the two
groups don't need to be the same size; unequal sizes only make rare words noisier, which Step 7 tests.
Matching does something undersampling can't: it makes the two groups alike on budget, genre and era, so what
remains different is closer to the script itself.

**What it measures.** Balance, as standardised mean differences (SMD): the gap between the groups' means in
units of their pooled standard deviation. |SMD| below 0.1 is the usual rule of thumb for "balanced".

**What it misses.** Matching only balances what we match on. Stars, studio, marketing and release date stay
unbalanced. With 67 unprofitable scripts, the matched sample is small."""),
    Run("s04_controls"),
    Fig("s04_controls", "balance", "Standardised mean differences between unprofitable and profitable scripts, "
        "before (triangles) and after (circles) matching. Points inside the dotted lines are balanced."),
    Table("s04_controls", "balance", "The same numbers as a table."),
    Reading("s04_controls"),
]
