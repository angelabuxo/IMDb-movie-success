"""Step 5: Scattertext. Term rates per 10,000 words in the successful vs unsuccessful scripts."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import s01_groups, s04_controls
from . import textstats as ts
from .config import cfg, log_run
from .paths import OUTPUTS, PRIVATE
from .report import HIGH, HIGH_SYMBOL, LOW, LOW_SYMBOL, NEUTRAL, Fig, Md, Reading, Run, Table, style

PRIMARY_TERMS = OUTPUTS / "s05_primary_terms.csv"
COMPARISON_TERMS = OUTPUTS / "s05_terms_{name}.csv"
GRID = OUTPUTS / "s05_grid.csv"
VARIANTS = OUTPUTS / "s05_variants.csv"
STRATA = OUTPUTS / "s05_strata_top.csv"
SUMMARY = OUTPUTS / "s05_summary.json"
EXPLORERS = PRIVATE / "scattertext"


def group_labels() -> dict[str, pd.Series]:
    sc = s01_groups.load()["scripts"].set_index("tconst")
    out = {k: sc[f"group_{k}"] for k in "ABC"}
    out["matched"] = s04_controls.labels()
    return out


def strata_labels() -> dict[str, pd.Series]:
    sc = s01_groups.load()["scripts"].set_index("tconst")
    st = s01_groups.tables(s01_groups.load())["strata"]
    out = {}
    for r in st[st.usable == "yes"].itertuples():
        col, key = ("group_C_genre", "primary_genre") if r.stratum == "genre" else ("group_C_decade", "decade_bin")
        out[f"{r.stratum}: {r.value}"] = sc[col].where(sc[key] == r.value)
    return out


def compare(labels: pd.Series, layer="dialogue", names="labels", stopwords="kept", lemmatize="off",
            min_df=None, mode="repeated"):
    p = cfg()["primary"]
    min_df = p["min_doc_freq"] if min_df is None else min_df
    X, vocab, ids, lengths = ts.dtm(layer, names, stopwords, lemmatize)
    Xs, ids_s, len_s, y = ts.comparison(X, ids, lengths, labels)
    res = ts.scores(Xs, len_s, y, min_df, mode)
    table = ts.term_table(vocab, res, Xs, y)
    thr = cfg()["scattertext"]["threshold"]
    meta = {"n_high": int(y.sum()), "n_low": int((y == 0).sum()), "terms": int(len(table)),
            "words_high": int(len_s[y == 1].sum()), "words_low": int(len_s[y == 0].sum()),
            "above_threshold": int((table.score.abs() >= thr).sum()),
            "above_high": int((table.score >= thr).sum()), "above_low": int((table.score <= -thr).sum())}
    return table, meta


def explorer(labels: pd.Series, name: str, **kw):
    """A Scattertext explorer from counts only: no document text goes in, so none can come out."""
    import scattertext as st
    p = cfg()["primary"]
    X, vocab, ids, lengths = ts.dtm(kw.get("layer", "dialogue"), kw.get("names", "labels"), "kept", "off")
    Xs, ids_s, _, y = ts.comparison(X, ids, lengths, labels)
    keep = np.asarray((Xs > 0).sum(axis=0)).ravel() >= p["min_doc_freq"]
    Xk = Xs[:, keep]
    voc = {t: i for i, t in enumerate(vocab[keep])}
    corpus = st.CorpusFromScikit(X=Xk, y=y, feature_vocabulary=voc,
                                 category_names=["unsuccessful", "successful"],
                                 raw_texts=[""] * Xk.shape[0]).build()
    html = st.produce_scattertext_explorer(corpus, category="successful", category_name="successful",
                                           not_category_name="unsuccessful", minimum_term_frequency=1,
                                           width_in_pixels=1000, use_full_doc=False,
                                           metadata=[str(t) for t in ids_s])
    EXPLORERS.mkdir(parents=True, exist_ok=True)
    path = EXPLORERS / f"{name}.html"
    path.write_text(html)
    return path


def run() -> dict:
    c = cfg()
    p = c["primary"]
    labels = group_labels()
    n = c["scattertext"]["top_n"]

    prim, meta = compare(labels[p["definition"]], p["layer"], "labels" if p["names_removed"] else "kept",
                         p["stopwords"], p["lemmatize"], p["min_doc_freq"], p["count_mode"])
    prim.to_csv(PRIMARY_TERMS, index=False)
    p_hi, p_lo = ts.top_terms(prim, n)

    for name in ("A", "B", "matched"):
        t, _ = compare(labels[name])
        t.to_csv(str(COMPARISON_TERMS).format(name=name), index=False)

    grid = []
    for name, lab in labels.items():
        for layer in c["text"]["layers"]:
            for min_df in c["scattertext"]["min_doc_freq_grid"]:
                for mode in c["scattertext"]["count_modes"]:
                    t, m = compare(lab, layer, "labels", "kept", "off", min_df, mode)
                    hi, lo = ts.top_terms(t, n)
                    grid.append({"definition": name, "layer": layer, "min_doc_freq": min_df, "count_mode": mode,
                                 **m, "jaccard_high_vs_primary": ts.jaccard(hi, p_hi),
                                 "jaccard_low_vs_primary": ts.jaccard(lo, p_lo),
                                 "is_primary": (name, layer, min_df, mode) == (p["definition"], p["layer"],
                                                                               p["min_doc_freq"], p["count_mode"])})
    pd.DataFrame(grid).to_csv(GRID, index=False)

    var = []
    for label, kw in (("primary", {}), ("names kept", dict(names="kept")),
                      ("names removed with NER too", dict(names="labels+ner")),
                      ("stopwords removed", dict(stopwords="removed")), ("lemmatised", dict(lemmatize="on"))):
        t, m = compare(labels[p["definition"]], **kw)
        hi, lo = ts.top_terms(t, n)
        var.append({"variant": label, **m, "jaccard_high_vs_primary": ts.jaccard(hi, p_hi),
                    "jaccard_low_vs_primary": ts.jaccard(lo, p_lo),
                    "top10_high": ", ".join(t.head(10).term), "top10_low": ", ".join(t.tail(10).term[::-1])})
    pd.DataFrame(var).to_csv(VARIANTS, index=False)

    strata = []
    for name, lab in strata_labels().items():
        t, m = compare(lab)
        strata.append({"stratum": name, **m, "top10_high": ", ".join(t.head(10).term),
                       "top10_low": ", ".join(t.tail(10).term[::-1]),
                       "jaccard_high_vs_primary": ts.jaccard(ts.top_terms(t, n)[0], p_hi),
                       "jaccard_low_vs_primary": ts.jaccard(ts.top_terms(t, n)[1], p_lo)})
    pd.DataFrame(strata).to_csv(STRATA, index=False)

    paths = [explorer(labels[k], f"dialogue_{k}") for k in ("C", "A", "B", "matched")]
    leak = check_explorers_for_text(paths)
    SUMMARY.write_text(json.dumps({"primary": meta, "explorers": [x.name for x in paths],
                                   "explorer_text_check": leak}, indent=1) + "\n")
    log_run("s05_scatter", [PRIMARY_TERMS.name, GRID.name, VARIANTS.name, STRATA.name, SUMMARY.name],
            {"primary": p, "scattertext": c["scattertext"]})
    return load()


def check_explorers_for_text(paths) -> dict:
    """Search each explorer for a long run of dialogue from the scripts it covers. With no text given to
    Scattertext there should be none; this checks rather than assumes."""
    from . import s02_layers
    lay = s02_layers.load_layers().set_index("tconst")
    probes = []
    for tid in lay.index[:40]:
        words = re.findall(r"[A-Za-z']+", lay.loc[tid, "dialogue"])
        for i in range(0, max(0, len(words) - 8), max(1, len(words) // 3)):
            probes.append(" ".join(words[i:i + 8]))
    out = {}
    for pth in paths:
        h = pth.read_text()
        out[pth.name] = int(sum(1 for pr in probes if pr in h))
    out["probes"] = len(probes)
    return out


def load() -> dict:
    return {"primary": pd.read_csv(PRIMARY_TERMS),
            "comparisons": {k: pd.read_csv(str(COMPARISON_TERMS).format(name=k)) for k in ("A", "B", "matched")},
            "grid": pd.read_csv(GRID), "variants": pd.read_csv(VARIANTS), "strata": pd.read_csv(STRATA),
            "summary": json.loads(SUMMARY.read_text())}


# ---------------------------------------------------------------- tables and figures

def tables(res: dict) -> dict:
    prim = res["primary"]
    n = 25
    top = pd.DataFrame({"more frequent in successful": prim.head(n).term.values,
                        "score (log10 ratio)": prim.head(n).score.round(2).values,
                        "scripts using it": prim.head(n).doc_freq.values,
                        "more frequent in unsuccessful": prim.tail(n).term.values[::-1],
                        "score": prim.tail(n).score.round(2).values[::-1],
                        "scripts": prim.tail(n).doc_freq.values[::-1]})
    g = res["grid"]
    stab = (g.groupby(["definition", "layer"])[["jaccard_high_vs_primary", "jaccard_low_vs_primary"]].mean()
            .round(2).reset_index())
    return {"top": top, "variants": res["variants"].round(3), "strata": res["strata"].round(3),
            "grid_summary": stab, "grid": g.round(3)}


def scatter(table: pd.DataFrame, title: str, n_labels: int = 8):
    e = cfg()["scattertext"]["smoothing_per_10k"]
    thr = cfg()["scattertext"]["threshold"]
    x, y = table.rate_low + e, table.rate_high + e
    side = np.where(table.score >= thr, "high", np.where(table.score <= -thr, "low", "neither"))
    fig = go.Figure()
    for s_, colour, sym, name in (("neither", NEUTRAL, "circle", "less than twice as frequent in either group"),
                                  ("high", HIGH, HIGH_SYMBOL, "≥ 2× more frequent in successful"),
                                  ("low", LOW, LOW_SYMBOL, "≥ 2× more frequent in unsuccessful")):
        m = side == s_
        fig.add_trace(go.Scattergl(x=x[m], y=y[m], mode="markers", name=name,
                                   marker=dict(color=colour, symbol=sym, size=5 if s_ == "neither" else 7,
                                               opacity=0.35 if s_ == "neither" else 0.85),
                                   text=table.term[m], customdata=table.doc_freq[m],
                                   hovertemplate="<b>%{text}</b><br>unsuccessful: %{x:.2f} per 10k<br>"
                                                 "successful: %{y:.2f} per 10k<br>scripts using it: %{customdata}<extra></extra>"))
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#444", dash="dash"),
                             name="same rate in both groups"))
    lab = pd.concat([table.head(n_labels), table.tail(n_labels)])
    fig.add_trace(go.Scatter(x=lab.rate_low + e, y=lab.rate_high + e, mode="text", text=lab.term,
                             textposition="top center", textfont=dict(size=10), showlegend=False, hoverinfo="skip"))
    fig.update_xaxes(type="log", title_text=f"rate in unsuccessful scripts, per 10,000 words (+{e}, log scale)")
    fig.update_yaxes(type="log", title_text=f"rate in successful scripts, per 10,000 words (+{e}, log scale)")
    return style(fig, title, height=620)


def figures(res: dict) -> dict:
    out = {"primary": scatter(res["primary"], "Primary: dialogue, (C) within genre × decade, names removed")}
    for k, label in (("A", "(A) gross below budget vs all others"), ("B", "(B) bottom vs top quartile, all films"),
                     ("matched", "Matched sample: unprofitable vs genre-, decade- and budget-matched profitable")):
        out[k] = scatter(res["comparisons"][k], f"Exploratory: {label}")
    g = res["grid"]
    piv = g.assign(run=g.layer + " · df≥" + g.min_doc_freq.astype(str) + " · " + g.count_mode)
    hm = piv.pivot_table(index="definition", columns="run",
                         values="jaccard_high_vs_primary").reindex(["C", "A", "B", "matched"])
    heat = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale="Cividis", zmin=0, zmax=1,
                                text=np.round(hm.values, 2), texttemplate="%{text}",
                                colorbar=dict(title="Jaccard")))
    heat.update_layout(xaxis_title="layer · minimum document frequency · count mode", yaxis_title="group definition")
    out["grid"] = style(heat, "Overlap of the top-50 'successful' terms with the primary run", height=420)
    return out


def reading(res: dict) -> str:
    m = res["summary"]["primary"]
    v = res["variants"].set_index("variant")
    g = res["grid"]
    nonprim = g[~g.is_primary]
    leak = res["summary"]["explorer_text_check"]
    return (f"The primary comparison has {m['n_high']} successful and {m['n_low']} unsuccessful scripts "
            f"({m['words_high']:,} and {m['words_low']:,} dialogue words) and {m['terms']:,} terms used by at least "
            f"{cfg()['primary']['min_doc_freq']} scripts. {m['above_threshold']} terms are at least twice as "
            f"frequent in one group ({m['above_high']} on the successful side, {m['above_low']} on the "
            f"unsuccessful side). Whether that is more than chance is Step 6's question; on its own this number "
            f"isn't a finding. Across the other {len(nonprim)} runs of the grid, the top-50 successful terms share "
            f"a median Jaccard overlap of {nonprim.jaccard_high_vs_primary.median():.2f} with the primary run "
            f"(unsuccessful side: {nonprim.jaccard_low_vs_primary.median():.2f}); 1 would mean identical lists. "
            f"Removing stopwords gives an overlap of {v.loc['stopwords removed', 'jaccard_high_vs_primary']:.2f}, "
            f"lemmatising {v.loc['lemmatised', 'jaccard_high_vs_primary']:.2f}, and keeping names "
            f"{v.loc['names kept', 'jaccard_high_vs_primary']:.2f}. The explorers were searched for "
            f"{leak['probes']} eight-word runs of dialogue and contain "
            f"{sum(v for k, v in leak.items() if k != 'probes')}.")


SECTION = [
    Md("""## Step 5 · Scattertext: which words does each group use comparatively more?

**What we do.** Each script is one document. For each group we add up how often every word is used and divide
by the group's total words, giving a **rate per 10,000 words**. We plot the unsuccessful group's rate (x)
against the successful group's rate (y) on log axes. The dashed diagonal is "same rate in both". Words near it
are shared language; words far above it are comparatively more frequent in successful scripts, words far
below it in unsuccessful ones. The distance from the diagonal is the score:

score = log10( (rate_successful + 0.1) / (rate_unsuccessful + 0.1) )

so a score of 0.3 means about twice as frequent. The 0.1 per 10,000 words is smoothing, so a word one group
never uses still gets a finite score.

**Why.** It is the course's two-corpus comparison (Scattertext, weeks 5-7). Using rates rather than counts
means the groups can differ in size and script length.

**Guarding against one script.** A word used 200 times in a single script can look distinctive by itself, so
a word must appear in at least **5 scripts** (document frequency, not total count). We also run 3 and 10, and
a "presence" mode that counts how many scripts use a word rather than how often.

**What it measures.** Which words each group uses comparatively more, in this corpus, under these
preprocessing choices.

**What it cannot tell us.** How a word is used (Step 8 reads the text), whether the difference is more than
chance (Step 6), whether it depends on which scripts are in the sample (Step 7), or anything about causes or
the quality of a script.

The primary run is definition (C), dialogue, names removed, minimum document frequency 5. All other runs are
exploratory."""),
    Run("s05_scatter"),
    Fig("s05_scatter", "primary", "Primary Scattertext: every dot is a word used by at least 5 scripts. x: rate in "
        "unsuccessful scripts, y: rate in successful scripts, both per 10,000 dialogue words on log scales. Words "
        "at least twice as frequent in one group are drawn as circles (successful) or triangles (unsuccessful); "
        "the labelled words are the most extreme on each side. Hover for counts. What to look for: whether the "
        "labelled words are plausible style or topic words or artefacts, which Step 8 checks in the text."),
    Table("s05_scatter", "top", "The 25 most extreme words on each side of the primary run, with the number of "
          "scripts using each."),
    Fig("s05_scatter", "A", "Exploratory: the same plot for definition (A). With 67 unsuccessful scripts, rare "
        "words on the unsuccessful side are noisy."),
    Fig("s05_scatter", "B", "Exploratory: definition (B), which mixes genres and eras; expect genre words."),
    Fig("s05_scatter", "matched", "Exploratory: the matched sample from Step 4 (same genre, decade and budget)."),
    Table("s05_scatter", "variants", "Preprocessing variants of the primary run: how many terms clear the "
          "2× threshold, and the Jaccard overlap of their top-50 lists with the primary run."),
    Fig("s05_scatter", "grid", "Jaccard overlap between each run's top-50 successful terms and the primary run's, "
        "across group definitions, layers, minimum document frequencies and count modes. Light = similar lists, "
        "dark = different."),
    Table("s05_scatter", "grid_summary", "Mean overlap with the primary run by definition and layer."),
    Table("s05_scatter", "strata", "Within single genres and decades where both groups have at least 30 scripts: "
          "the ten most extreme words on each side."),
    Reading("s05_scatter"),
    Md("""Interactive Scattertext explorers for (C), (A), (B) and the matched sample are written to
`analysis/scripts_text/private/scattertext/`. They are built from word counts only, with no document text, and
are checked for dialogue before use, but stay out of git as the brief asks."""),
]
