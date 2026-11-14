"""Step 7: does the result depend on which scripts happened to be in the sample?"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import s04_controls, s05_scatter, s06_null
from . import textstats as ts
from .config import cfg, log_run
from .paths import OUTPUTS
from .report import ACCENT, HIGH, HIGH_SYMBOL, LOW, LOW_SYMBOL, Fig, Md, Reading, Run, Table, style

SURVIVAL = OUTPUTS / "s07_survival.csv"
MATCHED = OUTPUTS / "s07_matched.csv"
SUMMARY = OUTPUTS / "s07_summary.json"
MATCH_ORDERS = 20


def run() -> dict:
    c = cfg()
    p = c["primary"]
    n = c["stability"]["top_n"]
    X, vocab, ids, lengths, y, _ = s06_null.primary_inputs()
    full = ts.term_table(vocab, ts.scores(X, lengths, y, p["min_doc_freq"], p["count_mode"]), X, y)
    f_hi, f_lo = full.head(n).term.tolist(), full.tail(n).term.tolist()[::-1]

    small = min(y.sum(), (y == 0).sum())
    big_label = 1 if y.sum() > (y == 0).sum() else 0
    big_idx, small_idx = np.where(y == big_label)[0], np.where(y != big_label)[0]
    seeds = [c["seed"] + 1000 + i for i in range(c["stability"]["subsamples"])]
    hits_hi, hits_lo, jac_hi, jac_lo = {t: 0 for t in f_hi}, {t: 0 for t in f_lo}, [], []
    for s_ in seeds:
        pick = np.sort(np.concatenate([small_idx, np.random.default_rng(s_).choice(big_idx, small, replace=False)]))
        t = ts.term_table(vocab, ts.scores(X[pick], lengths[pick], y[pick], p["min_doc_freq"], p["count_mode"]),
                          X[pick], y[pick])
        hi, lo = ts.top_terms(t, n)
        jac_hi.append(ts.jaccard(hi, set(f_hi)))
        jac_lo.append(ts.jaccard(lo, set(f_lo)))
        for w in f_hi:
            hits_hi[w] += w in hi
        for w in f_lo:
            hits_lo[w] += w in lo
    runs = len(seeds)
    surv = pd.DataFrame([{"side": "successful", "rank": i + 1, "term": w, "share_of_runs_in_top": hits_hi[w] / runs}
                         for i, w in enumerate(f_hi)] +
                        [{"side": "unsuccessful", "rank": i + 1, "term": w, "share_of_runs_in_top": hits_lo[w] / runs}
                         for i, w in enumerate(f_lo)])
    surv.to_csv(SURVIVAL, index=False)

    sc = s04_controls.frame()
    base = s05_scatter.compare(s04_controls.labels())[0]
    b_hi, b_lo = ts.top_terms(base, n)
    rows = []
    for cal in c["stability"]["matching_calipers"]:
        for k in range(MATCH_ORDERS):
            m = s04_controls.match(sc, cal, order_seed=c["seed"] + 5000 + k)
            lab = s04_controls.matched_labels(sc, m)
            t = s05_scatter.compare(lab)[0]
            hi, lo = ts.top_terms(t, n)
            rows.append({"caliper_sd": cal, "order_seed": c["seed"] + 5000 + k,
                         "pairs": int(m.control.notna().sum()), "jaccard_high": ts.jaccard(hi, b_hi),
                         "jaccard_low": ts.jaccard(lo, b_lo)})
    pd.DataFrame(rows).to_csv(MATCHED, index=False)

    summary = {"subsamples": runs, "seeds_first_last": [seeds[0], seeds[-1]], "subsample_size_per_group": int(small),
               "larger_group": "successful" if big_label == 1 else "unsuccessful", "top_n": n,
               "median_jaccard_high": float(np.median(jac_hi)), "median_jaccard_low": float(np.median(jac_lo)),
               "terms_in_top_in_90pct_high": int((surv[surv.side == "successful"].share_of_runs_in_top >= 0.9).sum()),
               "terms_in_top_in_90pct_low": int((surv[surv.side == "unsuccessful"].share_of_runs_in_top >= 0.9).sum()),
               "matched_median_jaccard_high": float(pd.DataFrame(rows).jaccard_high.median()),
               "matched_median_jaccard_low": float(pd.DataFrame(rows).jaccard_low.median())}
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s07_stability", [SURVIVAL.name, MATCHED.name, SUMMARY.name],
            {"stability": c["stability"], "seeds": seeds, "match_orders": MATCH_ORDERS})
    return load()


def load() -> dict:
    return {"survival": pd.read_csv(SURVIVAL), "matched": pd.read_csv(MATCHED),
            "summary": json.loads(SUMMARY.read_text())}


def tables(res: dict) -> dict:
    s = res["survival"]
    wide = pd.DataFrame({"rank": range(1, 21),
                         "successful term": s[s.side == "successful"].term.values[:20],
                         "share of runs": s[s.side == "successful"].share_of_runs_in_top.values[:20],
                         "unsuccessful term": s[s.side == "unsuccessful"].term.values[:20],
                         "share of runs ": s[s.side == "unsuccessful"].share_of_runs_in_top.values[:20]})
    m = res["matched"].groupby("caliper_sd").agg(pairs=("pairs", "median"),
                                                 jaccard_high=("jaccard_high", "median"),
                                                 jaccard_low=("jaccard_low", "median")).round(2).reset_index()
    return {"survival_top20": wide, "matched": m}


def figures(res: dict) -> dict:
    s = res["survival"]
    fig = go.Figure()
    for side, colour, sym in (("successful", HIGH, HIGH_SYMBOL), ("unsuccessful", LOW, LOW_SYMBOL)):
        v = s[s.side == side]
        fig.add_trace(go.Scatter(x=v["rank"], y=v.share_of_runs_in_top, mode="lines+markers", name=f"{side} side",
                                 text=v.term, line=dict(color=colour), marker=dict(symbol=sym, size=7),
                                 hovertemplate="<b>%{text}</b><br>rank %{x}<br>in top %{y:.0%} of runs<extra></extra>"))
    fig.update_layout(xaxis_title="rank of the term in the full-sample result",
                      yaxis_title=f"share of {res['summary']['subsamples']} subsamples where it stays in the top 50",
                      yaxis_tickformat=".0%")
    style(fig, "Survival of the full-sample top terms under subsampling", height=420)
    m = res["matched"]
    box = go.Figure()
    for side, colour in (("jaccard_high", HIGH), ("jaccard_low", LOW)):
        box.add_trace(go.Box(x=m.caliper_sd.astype(str) + " SD", y=m[side], name=side.replace("jaccard_", "") + " side",
                             marker_color=colour))
    box.update_layout(boxmode="group", xaxis_title="caliper on log budget",
                      yaxis_title="Jaccard overlap with the main matched result")
    style(box, "Matched design: different calipers and matching orders", height=400)
    return {"survival": fig, "matched": box}


def takeaway(res: dict) -> str:
    s = res["summary"]
    return (f"When we subsample the {s['larger_group']} group {s['subsamples']} times, {s['terms_in_top_in_90pct_high']} "
            f"successful-side and {s['terms_in_top_in_90pct_low']} unsuccessful-side terms stay in the top "
            f"{s['top_n']} in at least 90% of runs. The rest come and go with the sample.")


def reading(res: dict) -> str:
    s = res["summary"]
    return (f"We subsampled the {s['larger_group']} group down to {s['subsample_size_per_group']} scripts, "
            f"{s['subsamples']} times. The top-{s['top_n']} lists overlap the full-sample lists with a median "
            f"Jaccard of {s['median_jaccard_high']:.2f} on the successful side and {s['median_jaccard_low']:.2f} on "
            f"the unsuccessful side. {s['terms_in_top_in_90pct_high']} successful-side and "
            f"{s['terms_in_top_in_90pct_low']} unsuccessful-side terms stay in the top {s['top_n']} in at least 90% "
            f"of runs, and the rest come and go with the sample. In the matched design, changing the caliper and the "
            f"matching order gives a median overlap with the main matched result of "
            f"{s['matched_median_jaccard_high']:.2f} on the successful side and {s['matched_median_jaccard_low']:.2f} "
            f"on the unsuccessful side. The full-sample result stays the main one, and this is a robustness check.")


SECTION = [
    Md("""## Step 7 · Stability: one run is one sample, not the answer

This step asks whether the top word lists hold up when we change which scripts are in the sample.

### What we do

We draw 100 random subsamples of the larger group, each the size of the smaller group (seeds are logged in
`run_log.json`), and recompute the Scattertext each time. For each term in the full-sample top 50 we record
the share of runs in which it stays in the top 50. For the matched design we vary the caliper (0.25, 0.5 and 1
standard deviations of log budget) and the order in which scripts are matched (20 orders each).

### Why

A top-50 list from one sample can depend on which particular profitable scripts happened to be included. If a
term survives in most subsamples, it isn't carried by a few scripts. It's the same logic as running Louvain
with many seeds (week 4), where one run is one draw.

### What it measures

How sensitive the word lists are to the sample, as a survival share for each term and a Jaccard overlap for
each run (the words two lists share, divided by all words on either list).

### What it misses

Stable isn't the same as meaningful. A formatting artefact used by many scripts would be very stable. Step 8
reads the text."""),
    Run("s07_stability"),
    Fig("s07_stability", "survival", "For each term in the full-sample top 50 (x: its rank), the share of 100 "
        "subsamples in which it stays in the top 50 (y). A flat line near 100% would mean the list doesn't depend "
        "on the sample, and a fast drop means only the first few terms are robust."),
    Table("s07_stability", "survival_top20", "The 20 highest-ranked terms on each side and how often they survive."),
    Fig("s07_stability", "matched", "For the matched design, the overlap of the top-50 lists with the main matched "
        "result, across calipers and 20 matching orders per caliper."),
    Table("s07_stability", "matched", "Median pairs and overlaps for each caliper."),
    Reading("s07_stability"),
]
