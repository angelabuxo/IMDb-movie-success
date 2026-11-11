"""Step 10: franchise and leakage checks."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import s01_groups, s02_layers, s05_scatter, s06_null
from . import textstats as ts
from .config import cfg, log_run
from .paths import FEATURES, OUTPUTS
from .report import HIGH, HIGH_SYMBOL, LOW, LOW_SYMBOL, Fig, Md, Reading, Run, Table, style

FLAGS = OUTPUTS / "s10_flags.csv"
RESULTS = OUTPUTS / "s10_results.csv"
RANDOM_DROP = OUTPUTS / "s10_random_drop.csv"
RANDOM_DROPS = 50     # exploratory control: drop as many random non-franchise films instead
SUMMARY = OUTPUTS / "s10_summary.json"
NULL_SHUFFLES = 200


def _ids(path, col=0):
    t = pd.read_csv(path, sep="\t", header=0, quoting=3, dtype=str, on_bad_lines="skip")
    return set(t.iloc[:, col].str.strip('"'))


def flags() -> pd.DataFrame:
    sc = s01_groups.load()["scripts"]
    rep = s02_layers.load()["report"].set_index("tconst")
    follows, franchise, series = (_ids(FEATURES / f) for f in ("wd_follows.tsv", "wd_franchise.tsv", "wd_series.tsv"))
    f = pd.DataFrame({"tconst": sc.tconst, "movie_name": sc.movie_name})
    f["sequel_follows"] = f.tconst.isin(follows)
    f["named_franchise"] = f.tconst.isin(franchise)
    f["series_noisy"] = f.tconst.isin(series)
    f["likely_franchise"] = f.sequel_follows | f.named_franchise
    # A post-release transcript has dialogue but almost no description or scene headings.
    r = rep.loc[f.tconst]
    f["action_share"] = (r.action_words / r.full_words).to_numpy()
    f["n_scenes"] = r.n_scenes.to_numpy()
    f["possible_transcript"] = (f.action_share < 0.15) | (f.n_scenes < cfg()["text"]["min_scenes_flag"])
    return f


def count_test(X, ids, lengths, lab, sc, seed, shuffles=NULL_SHUFFLES):
    """The primary count test (terms at least 2x in one group) against a stratified null, for one labelling."""
    c, p = cfg(), cfg()["primary"]
    Xs, ids_s, len_s, y = ts.comparison(X, ids, lengths, lab)
    strata = sc.loc[ids_s, "cell"].to_numpy()
    nl = ts.shuffle_null(Xs, len_s, y, strata, p["min_doc_freq"], shuffles, seed, p["count_mode"], progress=False)
    real = int((np.abs(nl["real"]["score"][nl["real"]["ok"]]) >= c["scattertext"]["threshold"]).sum())
    sd = nl["n_above"].std(ddof=1)
    return {"successful": int(y.sum()), "unsuccessful": int((y == 0).sum()), "terms_above_threshold": real,
            "null_mean": float(nl["n_above"].mean()), "null_std": float(sd),
            "z": float((real - nl["n_above"].mean()) / sd),
            "p_empirical": float(ts.empirical_p((nl["n_above"] >= real).sum(), shuffles))}


def run() -> dict:
    c = cfg()
    p = c["primary"]
    n = c["scattertext"]["top_n"]
    f = flags()
    f.to_csv(FLAGS, index=False)
    sc = s01_groups.load()["scripts"].set_index("tconst")
    base = sc[f"group_{p['definition']}"]
    prim = s05_scatter.load()["primary"]
    p_hi, p_lo = ts.top_terms(prim, n)
    X, vocab, ids, lengths = ts.dtm(p["layer"], "labels", p["stopwords"], p["lemmatize"])
    rows = []
    for label, drop in (("primary", set()), ("without likely franchise films", set(f.tconst[f.likely_franchise])),
                        ("without sequels only", set(f.tconst[f.sequel_follows])),
                        ("without possible transcripts", set(f.tconst[f.possible_transcript]))):
        lab = base.copy()
        lab[lab.index.isin(drop)] = None
        Xs, ids_s, len_s, y = ts.comparison(X, ids, lengths, lab)
        t = ts.term_table(vocab, ts.scores(Xs, len_s, y, p["min_doc_freq"], p["count_mode"]), Xs, y)
        hi, lo = ts.top_terms(t, n)
        strata = sc.loc[ids_s, "cell"].to_numpy()
        nl = ts.shuffle_null(Xs, len_s, y, strata, p["min_doc_freq"], NULL_SHUFFLES, c["seed"] + 11,
                             p["count_mode"], progress=False)
        real = int((np.abs(nl["real"]["score"][nl["real"]["ok"]]) >= c["scattertext"]["threshold"]).sum())
        in_comparison = set(base.index[base.isin(["low", "high"])]) - ts.excluded_scripts()
        rows.append({"run": label, "dropped": len(drop & in_comparison),
                     "successful": int(y.sum()), "unsuccessful": int((y == 0).sum()),
                     "jaccard_high_vs_primary": ts.jaccard(hi, p_hi), "jaccard_low_vs_primary": ts.jaccard(lo, p_lo),
                     "terms_above_threshold": real,
                     "null_mean": float(nl["n_above"].mean()), "null_std": float(nl["n_above"].std(ddof=1)),
                     "z": float((real - nl["n_above"].mean()) / nl["n_above"].std(ddof=1)),
                     "p_empirical": float(ts.empirical_p((nl["n_above"] >= real).sum(), NULL_SHUFFLES)),
                     "top10_high": ", ".join(t.head(10).term)})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS, index=False)

    # Control: dropping 93 films also shrinks the groups and costs power. Drop the same number from each
    # group at random among the non-franchise films instead, 50 times, each against its own null.
    comp = set(base.index[base.isin(["low", "high"])]) - ts.excluded_scripts()
    fr = set(f.tconst[f.likely_franchise]) & comp
    n_hi = sum(base[t] == "high" for t in fr)
    n_lo = sum(base[t] == "low" for t in fr)
    pool_hi = sorted(t for t in comp - fr if base[t] == "high")
    pool_lo = sorted(t for t in comp - fr if base[t] == "low")
    rng = np.random.default_rng(c["seed"] + 21)
    drops = []
    for k in range(RANDOM_DROPS):
        drop = set(rng.choice(pool_hi, n_hi, replace=False)) | set(rng.choice(pool_lo, n_lo, replace=False))
        lab = base.copy()
        lab[lab.index.isin(drop)] = None
        drops.append({"replicate": k + 1, **count_test(X, ids, lengths, lab, sc, c["seed"] + 100 + k)})
    pd.DataFrame(drops).to_csv(RANDOM_DROP, index=False)

    # The classifier of Step 9 without the likely franchise films (3 repeats, C fixed at Step 9's choice).
    from . import s09_logreg
    lab = base.copy()
    lab[lab.index.isin(set(f.tconst[f.likely_franchise]))] = None
    Xs, ids_s, len_s, y = ts.comparison(X, ids, lengths, lab)
    meta = sc.loc[ids_s]
    B = s09_logreg.baseline_matrix(meta)
    strat = np.array([f"{g}|{l}" for g, l in zip(meta.primary_genre, y)])
    C_fixed = s09_logreg.load()["summary"]["C_chosen_median"]
    cvr = pd.DataFrame([r_ for k in range(3) for r_ in s09_logreg.cv_once(Xs, len_s, y, B, strat, c["seed"] + k, C=C_fixed)])
    logreg_nf = {k: float(cvr[k].mean()) for k in ("auc_text", "auc_baseline", "auc_text_plus_baseline")}
    lab_films = sc.index[base.isin(["low", "high"]) & ~sc.index.isin(ts.excluded_scripts())]
    ff = f.set_index("tconst").loc[lab_films]
    summary = {"labelled_films": int(len(lab_films)), "likely_franchise": int(ff.likely_franchise.sum()),
               "sequel_follows": int(ff.sequel_follows.sum()), "named_franchise": int(ff.named_franchise.sum()),
               "series_noisy": int(ff.series_noisy.sum()), "possible_transcripts": int(ff.possible_transcript.sum()),
               "franchise_share_successful": float(ff[base[lab_films] == "high"].likely_franchise.mean()),
               "franchise_share_unsuccessful": float(ff[base[lab_films] == "low"].likely_franchise.mean()),
               "null_shuffles": NULL_SHUFFLES, "logreg_without_franchise": logreg_nf,
               "random_drop": {"replicates": RANDOM_DROPS, "dropped_successful": int(n_hi),
                               "dropped_unsuccessful": int(n_lo),
                               "z_median": float(np.median([d["z"] for d in drops])),
                               "z_p05": float(np.percentile([d["z"] for d in drops], 5)),
                               "z_p95": float(np.percentile([d["z"] for d in drops], 95)),
                               "share_p_below_0_05": float(np.mean([d["p_empirical"] <= 0.05 for d in drops])),
                               "share_z_at_or_below_franchise_drop": None}}
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    z_fr = float(res.set_index("run").loc["without likely franchise films", "z"])
    summary["random_drop"]["share_z_at_or_below_franchise_drop"] = float(np.mean([d["z"] <= z_fr for d in drops]))
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s10_franchise", [FLAGS.name, RESULTS.name, RANDOM_DROP.name, SUMMARY.name],
            {"null_shuffles": NULL_SHUFFLES, "random_drops": RANDOM_DROPS})
    return load()


def load() -> dict:
    return {"flags": pd.read_csv(FLAGS, dtype={"tconst": str}), "results": pd.read_csv(RESULTS),
            "random_drop": pd.read_csv(RANDOM_DROP),
            "summary": json.loads(SUMMARY.read_text())}


def tables(res: dict) -> dict:
    r = res["results"].copy()
    num = ["jaccard_high_vs_primary", "jaccard_low_vs_primary", "null_mean", "null_std", "z", "p_empirical"]
    r[num] = r[num].round(3)
    return {"results": r}


def figures(res: dict) -> dict:
    r = res["results"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r.run, y=r.jaccard_high_vs_primary, name="successful side", marker_color=HIGH))
    fig.add_trace(go.Bar(x=r.run, y=r.jaccard_low_vs_primary, name="unsuccessful side", marker_color=LOW,
                         marker_pattern_shape="/"))
    fig.update_layout(barmode="group", yaxis_title="Jaccard overlap of the top-50 lists with the primary run",
                      yaxis_range=[0, 1])
    z = go.Figure()
    z.add_hrect(y0=-1.96, y1=1.96, fillcolor="#999999", opacity=0.15, line_width=0,
                annotation_text="grey band: |z| < 1.96, compatible with chance", annotation_position="bottom left")
    short = {"primary": "all films", "without likely franchise films": "no franchise films",
             "without sequels only": "no sequels", "without possible transcripts": "no possible transcripts"}
    z.add_trace(go.Scatter(x=r.run.map(short), y=r.z, mode="markers+text", text=[f"z = {v:.2f}<br>p = {p_:.3f}" for v, p_ in
                                                                      zip(r.z, r.p_empirical)],
                           textposition="middle right", marker=dict(size=14, color=HIGH, symbol=HIGH_SYMBOL),
                           showlegend=False))
    rd = res["random_drop"]
    z.add_trace(go.Box(x=["random drops (control)"] * len(rd), y=rd.z, name="random drops",
                       marker_color=LOW, boxpoints="all", jitter=0.4, pointpos=0, showlegend=False))
    z.update_layout(yaxis_title="z against its own stratified null",
                    xaxis_title="which films are in the comparison")
    return {"overlap": style(fig, "Do the top terms survive without franchise films and transcripts?", height=400),
            "z": style(z, "The count test with and without franchise films", height=420)}


def reading(res: dict) -> str:
    s, r = res["summary"], res["results"].set_index("run")
    nf = r.loc["without likely franchise films"]
    nt = r.loc["without possible transcripts"]
    rd = s["random_drop"]
    return (f"Of the {s['labelled_films']} films in the primary comparison, {s['likely_franchise']} are likely "
            f"franchise films ({s['sequel_follows']} follow another film on Wikidata, {s['named_franchise']} belong "
            f"to a named media franchise); they make up {s['franchise_share_successful']:.0%} of the successful and "
            f"{s['franchise_share_unsuccessful']:.0%} of the unsuccessful group. Dropping them, the top-50 lists "
            f"overlap the primary ones with Jaccard {nf.jaccard_high_vs_primary:.2f} (successful) and "
            f"{nf.jaccard_low_vs_primary:.2f} (unsuccessful), and the count of terms above 2× has z = {nf.z:.2f} "
            f"against {s['null_shuffles']} stratified shuffles (p = {nf.p_empirical:.3f}). As an exploratory control, "
            f"dropping the same number of films ({rd['dropped_successful']} successful, {rd['dropped_unsuccessful']} "
            f"unsuccessful) at random among the non-franchise films, {rd['replicates']} times, gives a median z of "
            f"{rd['z_median']:.2f} (90% between {rd['z_p05']:.2f} and {rd['z_p95']:.2f}); "
            f"{rd['share_z_at_or_below_franchise_drop']:.0%} of random drops fall as low as dropping the franchise "
            f"films does, so " + ("losing those particular films, not just losing films, is what removes the excess."
            if rd["share_z_at_or_below_franchise_drop"] < 0.1 else "part of the drop is simply lost power.")
            + f" {s['possible_transcripts']} "
            f"scripts look like they could be transcripts (less than 15% action words, or fewer than "
            f"{cfg()['text']['min_scenes_flag']} scenes); without them the overlap is "
            f"{nt.jaccard_high_vs_primary:.2f} and z = {nt.z:.2f}. Without franchise films, Step 9's classifier gives "
            f"AUC {s['logreg_without_franchise']['auc_text']:.3f} for text against "
            f"{s['logreg_without_franchise']['auc_baseline']:.3f} for the baseline (3 repeats). The Wikidata series "
            f"property is reported but "
            f"not used: it marks {s['series_noisy']} of these films and includes lists, not just series.")


SECTION = [
    Md("""## Step 10 · Franchise and leakage checks

**What we do.** We flag likely franchise films two ways: Wikidata's "follows" property (the film is a sequel;
clean, but it misses first films) and membership of a named media franchise (catches first films too). The
"series" property is reported but not used, because it also holds lists such as "100 greatest films". We
rerun the primary comparison without franchise films, without sequels only, and without scripts that might be
post-release transcripts (almost no action description, or fewer than 20 scenes), each with its own
200-shuffle null.

**Why.** Sequels reuse characters, places and catchphrases, and franchise films are often hits. If a few
franchises drive the "successful" vocabulary, the result is about franchises, not scripts. Name removal works
script by script, so a sequel's script can still contain a character name that is only a speaker in the
first film. A transcript written from the finished film is not the script that was green-lit, which matters
for anything we say about scripts predicting success.

**What it measures.** Whether the top terms and the overall count test survive without these films.

**What it misses.** Both franchise flags are incomplete, and the transcript rule is a heuristic."""),
    Run("s10_franchise"),
    Fig("s10_franchise", "z", "z of the primary count test (terms at least twice as frequent in one group) "
        "against its own 200-shuffle stratified null, for the full comparison and with each set of films "
        "removed. The box (exploratory control) drops the same number of random non-franchise films 50 times. "
        "Inside the grey band the real groups are no more different than random ones."),
    Fig("s10_franchise", "overlap", "Overlap of the top-50 lists with the primary run after dropping each set of "
        "films. Bars near 1: the result doesn't depend on those films."),
    Table("s10_franchise", "results", "Each rerun: films dropped, group sizes, overlap with the primary run, and "
          "the count test against its own stratified null."),
    Reading("s10_franchise"),
]
