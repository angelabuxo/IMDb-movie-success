"""Step 6: compared to what? The primary Scattertext against 1,000 stratified label shuffles."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import s01_groups
from . import textstats as ts
from .config import cfg, log_run
from .paths import OUTPUTS
from .report import ACCENT, HIGH, LOW, NEUTRAL, Fig, Md, Reading, Run, Table, style

NULL_COUNTS = OUTPUTS / "s06_null_counts.csv"
TERM_NULL = OUTPUTS / "s06_term_null.csv"
SUMMARY = OUTPUTS / "s06_summary.json"


def primary_inputs():
    p = cfg()["primary"]
    sc = s01_groups.load()["scripts"].set_index("tconst")
    labels = sc[f"group_{p['definition']}"]
    X, vocab, ids, lengths = ts.dtm(p["layer"], "labels" if p["names_removed"] else "kept", p["stopwords"], p["lemmatize"])
    Xs, ids_s, len_s, y = ts.comparison(X, ids, lengths, labels)
    strata = sc.loc[ids_s, "cell"].to_numpy()
    return Xs, vocab, ids_s, len_s, y, strata


def _test(real, null):
    """z-score and one-sided empirical p of a real count against its null."""
    mu, sd = float(np.mean(null)), float(np.std(null, ddof=1))
    return {"real": int(real), "null_mean": mu, "null_std": sd, "z": (real - mu) / sd if sd > 0 else None,
            "p": float(ts.empirical_p((np.asarray(null) >= real).sum(), len(null))),
            "null_p05": float(np.percentile(null, 5)), "null_p95": float(np.percentile(null, 95))}


def run() -> dict:
    c = cfg()
    p = c["primary"]
    thr = c["scattertext"]["threshold"]
    X, vocab, ids, lengths, y, strata = primary_inputs()
    out = ts.shuffle_null(X, lengths, y, strata if c["shuffle_null"]["stratified"] else None, p["min_doc_freq"],
                          c["shuffle_null"]["shuffles"], c["seed"], p["count_mode"])
    glob = ts.shuffle_null(X, lengths, y, None, p["min_doc_freq"], c["shuffle_null"]["shuffles"], c["seed"] + 1,
                           p["count_mode"], progress=False)
    ok = out["real"]["ok"]
    rs = out["real"]["score"][ok]
    real_above = int((np.abs(rs) >= thr).sum())
    pd.DataFrame({"shuffle": np.arange(1, out["shuffles"] + 1), "terms_above_threshold": out["n_above"],
                  "above_successful_side": out["n_above_hi"], "above_unsuccessful_side": out["n_above_lo"],
                  "max_abs_score": out["max_abs"], "terms_above_threshold_global_shuffle": glob["n_above"]}
                 ).to_csv(NULL_COUNTS, index=False)

    p_term = ts.empirical_p(out["extreme"], out["shuffles"])
    p_fwer = ts.empirical_p((out["max_abs"][None, :] >= np.abs(rs)[:, None]).sum(axis=1), out["shuffles"])
    t = pd.DataFrame({"term": vocab[ok], "doc_freq": out["real"]["df"][ok], "score": rs,
                      "null_mean": out["mean"], "null_std": out["std"],
                      "z": (rs - out["mean"]) / np.where(out["std"] > 0, out["std"], np.nan),
                      "p_empirical": p_term, "q_bh": ts.bh(p_term), "p_familywise": p_fwer})
    t = t.sort_values("score", ascending=False)
    t.to_csv(TERM_NULL, index=False)

    summary = {
        "shuffles": out["shuffles"], "stratified": c["shuffle_null"]["stratified"], "strata": int(len(np.unique(strata))),
        "n_high": int(y.sum()), "n_low": int((y == 0).sum()), "terms": int(ok.sum()),
        "count_test": _test(real_above, out["n_above"]),
        "count_test_successful_side": _test(int((rs >= thr).sum()), out["n_above_hi"]),
        "count_test_unsuccessful_side": _test(int((rs <= -thr).sum()), out["n_above_lo"]),
        "count_test_global_shuffle": _test(real_above, glob["n_above"]),
        "terms_q_below_0_05": int((t.q_bh < 0.05).sum()),
        "terms_familywise_below_0_05": int((t.p_familywise < 0.05).sum()),
        "smallest_p_possible": float(1 / (1 + out["shuffles"])),
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s06_null", [NULL_COUNTS.name, TERM_NULL.name, SUMMARY.name],
            {"null": c["shuffle_null"], "primary": p, "threshold": thr, "seed": c["seed"]})
    return load()


def load() -> dict:
    return {"counts": pd.read_csv(NULL_COUNTS), "terms": pd.read_csv(TERM_NULL),
            "summary": json.loads(SUMMARY.read_text())}


def tables(res: dict) -> dict:
    s = res["summary"]
    rows = []
    for key, label in (("count_test", "all terms, stratified shuffle (primary)"),
                       ("count_test_successful_side", "successful side, stratified"),
                       ("count_test_unsuccessful_side", "unsuccessful side, stratified"),
                       ("count_test_global_shuffle", "all terms, unstratified shuffle (secondary)")):
        v = s[key]
        rows.append({"test": label, "real count": v["real"], "null mean": round(v["null_mean"], 1),
                     "null std": round(v["null_std"], 1), "null 5th-95th pct": f"{v['null_p05']:.0f}–{v['null_p95']:.0f}",
                     "z": round(v["z"], 2) if v["z"] is not None else None, "empirical p": round(v["p"], 4)})
    t = res["terms"]
    top = pd.concat([t.head(15), t.tail(15)])[["term", "doc_freq", "score", "z", "p_empirical", "q_bh", "p_familywise"]]
    return {"count_tests": pd.DataFrame(rows), "top_terms": top.round(4)}


def figures(res: dict) -> dict:
    s, cnt = res["summary"], res["counts"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=cnt.terms_above_threshold, name="stratified shuffles", marker_color=NEUTRAL,
                               opacity=0.8, nbinsx=40))
    fig.add_trace(go.Histogram(x=cnt.terms_above_threshold_global_shuffle, name="unstratified shuffles (outline)",
                               marker=dict(color="rgba(0,0,0,0)", line=dict(color=ACCENT, width=1.5)), nbinsx=40))
    real = s["count_test"]["real"]
    fig.add_vline(x=real, line_color=LOW, line_width=4, layer="above",
                  annotation_text=f"real: {real} (z = {s['count_test']['z']:.1f}, p = {s['count_test']['p']:.3f})",
                  annotation_position="top")
    fig.update_layout(barmode="overlay", xaxis_title="number of terms at least twice as frequent in one group",
                      yaxis_title=f"shuffles (of {s['shuffles']:,})")
    style(fig, "The real count against 1,000 label shuffles", height=420)

    t = res["terms"]
    zf = go.Figure(go.Scattergl(x=t.score, y=t.z, mode="markers", text=t.term,
                                marker=dict(size=5, color=np.where(t.q_bh < 0.05, LOW, NEUTRAL)),
                                hovertemplate="<b>%{text}</b><br>score %{x:.2f}<br>z %{y:.1f}<extra></extra>"))
    zf.update_layout(xaxis_title="score: log10 ratio of rates (successful / unsuccessful)",
                     yaxis_title="z against that term's own shuffle distribution")
    n_sig = int((t.q_bh < 0.05).sum())
    style(zf, "Each term's score against its own null " + (f"(orange: BH q < 0.05, {n_sig} terms)" if n_sig
          else "(no term passes BH q < 0.05)"), height=420)
    return {"count_null": fig, "term_z": zf}


def takeaway(res: dict) -> str:
    s = res["summary"]
    ct = s["count_test"]
    if ct["p"] > 0.05:
        head = (f"There's no evidence that successful and unsuccessful scripts differ in vocabulary more than random "
                f"groupings within the same genre and decade do (z = {ct['z']:.2f}, p = {ct['p']:.3f}).")
    elif ct["p"] > 0.01:
        head = (f"There's weak evidence that successful and unsuccessful scripts differ in vocabulary more than "
                f"random groupings within the same genre and decade do (z = {ct['z']:.2f}, p = {ct['p']:.3f}).")
    else:
        head = (f"Successful and unsuccessful scripts differ in vocabulary more than random groupings within the "
                f"same genre and decade do (z = {ct['z']:.2f}, p = {ct['p']:.3f}).")
    return (f"{head} Word by word, {s['terms_q_below_0_05']} terms survive the Benjamini-Hochberg correction at "
            f"q < 0.05.")


def reading(res: dict) -> str:
    s = res["summary"]
    ct = s["count_test"]
    hi, lo = s["count_test_successful_side"], s["count_test_unsuccessful_side"]
    if ct["p"] > 0.05:
        verdict = ("The real count sits inside the null distribution. There's **no evidence** that successful and "
                   "unsuccessful scripts differ in vocabulary more than random groupings of the same scripts would, "
                   "once genre and decade are held fixed.")
    elif ct["p"] > 0.01:
        verdict = (f"The real count sits at the upper edge of the null distribution, which is **weak evidence** that "
                   f"the two groups differ in vocabulary more than random groupings within the same genre and decade "
                   f"cells. The excess is on the successful side (z = {hi['z']:.2f}, p = {hi['p']:.3f}), while the "
                   f"unsuccessful side looks like chance (z = {lo['z']:.2f}, p = {lo['p']:.3f}). It's one test among "
                   f"several in these notebooks, and Steps 8 and 10 check what the extra terms are.")
    else:
        verdict = ("The real count sits well above the null distribution, so the two groups differ in vocabulary more "
                   "than random groupings of the same scripts within the same genre and decade cells.")
    return (f"In the real comparison, {ct['real']} terms are at least twice as frequent in one group. Across "
            f"{s['shuffles']:,} label shuffles within {s['strata']} genre × decade cells, that count averages "
            f"{ct['null_mean']:.1f} (standard deviation {ct['null_std']:.1f}, and 90% of shuffles fall between "
            f"{ct['null_p05']:.0f} and {ct['null_p95']:.0f}). That gives z = {ct['z']:.2f} and an empirical p = "
            f"{ct['p']:.3f}. {verdict} Many terms clear the threshold in every shuffle, because with "
            f"{s['terms']:,} terms and a few hundred scripts per group, rare words cross 2× by chance. Term by term, "
            f"{s['terms_q_below_0_05']} pass a Benjamini-Hochberg q < 0.05 and {s['terms_familywise_below_0_05']} "
            f"beat the null of the largest score at p < 0.05 (with {s['shuffles']:,} shuffles the smallest possible p "
            f"is {s['smallest_p_possible']:.4f}). A single real value isn't a finding. It only means something next "
            f"to this histogram.")


SECTION = [
    Md("""## Step 6 · Compared to what? A shuffle null

This step asks whether the real groups differ in vocabulary more than random groups of the same scripts
would.

### What we do

We keep every script and its words, but shuffle the success labels 1,000 times and redo the Scattertext each
time. The shuffle is **stratified**, meaning labels are only swapped between films in the same genre × decade
cell, so every shuffled world has the same genre and era mix in each group as the real one. For each shuffle
we record (i) every term's score and (ii) how many terms clear the 2× threshold. Together the shuffles give
the null distribution, the spread of results we'd see if the labels meant nothing.

### Why

Step 5 always finds "distinctive" words, because with thousands of words and a few hundred scripts some will
be twice as frequent in one group by luck. The question is whether the real comparison finds *more*, or more
extreme, differences than random groupings do. This is the course's "compared to what?" (weeks 2-4, null
models).

### What it measures

For the count, z = (real − mean of shuffles) / standard deviation of shuffles, and the empirical p = (1 +
number of shuffles at least as extreme) / (1 + 1,000), which is never exactly 0. For each term we do the same
against that term's own shuffle distribution. Since that means testing thousands of terms, we correct for it
two ways. Benjamini-Hochberg limits the expected share of false positives among the terms we flag, and a
stricter family-wise test compares each term with the largest score in each shuffle.

### What it misses

The null keeps genre and decade fixed but not budget, stars or anything else. It tests whether the
vocabularies differ, not why."""),
    Run("s06_null"),
    Fig("s06_null", "count_null", "The grey histogram counts, for each of 1,000 stratified shuffles, the terms "
        "at least twice as frequent in one group. The green outline shows the same under unstratified shuffles, "
        "and the orange line is the real comparison. If the line sits inside the grey histogram, the real groups "
        "are no more different than random ones."),
    Table("s06_null", "count_tests", "The count test overall and for each side, with the unstratified shuffle "
          "for comparison."),
    Fig("s06_null", "term_z", "Every term's real score (x) against its z-score within its own shuffle "
        "distribution (y). Terms in orange survive the Benjamini-Hochberg correction at q < 0.05."),
    Table("s06_null", "top_terms", "The 15 most extreme terms on each side, with their null statistics."),
    Reading("s06_null"),
]
