"""Tables, figures and plain-language readings of the saved outputs. The notebook and the explorer page both
call these, so their numbers can't differ. Everything here reads outputs/ only, except film_map(), which
needs the vectors in data/raw and returns None without them.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .paths import OUTPUTS, PRIVATE, VECTORS, film_table

LOW, HIGH, NEUTRAL, ACCENT = "#D55E00", "#0072B2", "#999999", "#009E73"
FONT = dict(family="Inter, Helvetica, Arial, sans-serif", size=13)
CORPORA = ["reviews", "screenplays", "summaries"]
LABEL = {"reviews": "Audience reviews", "screenplays": "Screenplays", "summaries": "Plot summaries"}
TARGET = {"log_mult": "Profitability<br>ln(gross / budget)", "log_gross": "Revenue<br>ln(gross)",
          "log_budget": "Budget<br>ln(budget)"}
ALL = "all clean films"
PRIMARY = "net_of_franchise_too"   # budget, year, genre and the franchise flag (post hoc, see config.yaml)


def style(fig, title, height=420):
    fig.update_layout(template="simple_white", font=FONT, height=height, margin=dict(l=70, r=30, t=90, b=60),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      title=dict(text=title, x=0.01, xanchor="left"))
    return fig


def load() -> dict:
    return {"results": pd.read_csv(OUTPUTS / "s03_results.csv"),
            "nulls": json.loads((OUTPUTS / "s03_nulls.json").read_text()),
            "words": pd.read_csv(OUTPUTS / "s03_words.csv"),
            "summary": json.loads((OUTPUTS / "s03_summary.json").read_text()),
            "cleaning": json.loads((OUTPUTS / "s01_cleaning.json").read_text()),
            "embedding": json.loads((OUTPUTS / "s02_embedding.json").read_text())}


def pick(r: pd.DataFrame, **kw) -> pd.DataFrame:
    kw = {"sample": ALL, "representation": "embedding", "variant": "clean", **kw}
    m = np.ones(len(r), bool)
    for k, v in kw.items():
        if v is not None:
            m &= (r[k] == v).to_numpy()
    return r[m]


def row(r: pd.DataFrame, corpus: str, analysis: str, target: str = "log_mult", **kw) -> pd.Series:
    got = pick(r, corpus=corpus, analysis=analysis, target=target, **kw)
    assert len(got) == 1, (corpus, analysis, target, kw, len(got))
    return got.iloc[0]


# ---------------------------------------------------------------- tables

def cleaning_table(c: dict) -> pd.DataFrame:
    s, m, v = c["screenplays"], c["summaries"], c["reviews"]
    rows = [
        ("Audience reviews", v["films"], f"{v['reviews_sampled']:,} reviews, {v['median_reviews_per_film']:.0f} per film (median)",
         f"{v['dropped_duplicate']:,} duplicates, {v['dropped_empty_or_short']:,} too short and {v['dropped_not_english']:,} "
         f"non-English reviews dropped before sampling; {v['share_of_sentences_dropped_for_money']:.1%} of sentences "
         f"dropped for money talk (in {v['sampled_reviews_with_a_money_sentence']:,} reviews); "
         f"{v['title_mentions_masked']:,} title and {v['cast_and_director_mentions_masked']:,} cast or director mentions masked; "
         f"{v.get('other_film_title_mentions_masked', 0):,} mentions of other films' titles and "
         f"{v.get('other_people_mentions_masked', 0):,} of other people masked"),
        ("Screenplays", s["films"], f"{s['median_words_clean']:,.0f} words per script (median)",
         f"{s['share_of_raw_words_removed']:.1%} of words removed (headings, speaker labels, parentheticals, "
         f"transitions); {s['median_distinct_names_masked']:.0f} character names masked per script (median), "
         f"{s['share_of_clean_words_that_are_name_placeholders']:.1%} of the words left"),
        ("Plot summaries", m["films"], f"{m['median_words_clean']:,.0f} words per summary (median)",
         f"{m['removed']['people']:,} person names masked with spaCy NER "
         f"({m['share_of_clean_words_that_are_name_placeholders']:.1%} of words), {m['removed']['actor_parens']:,} "
         f"actor names in brackets and {m['removed']['titles']:,} title mentions removed; "
         f"{m['dropped_too_short']} summaries under 100 words dropped"),
    ]
    return pd.DataFrame(rows, columns=["corpus", "films", "text kept", "what cleaning did"])


def main_table(r: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in CORPORA:
        for variant in ("raw", "clean"):
            a = {t: row(r, c, "alone", t, variant=variant) for t in TARGET}
            n0 = row(r, c, "net", variant=variant)
            n = row(r, c, PRIMARY, variant=variant)
            out.append({"corpus": LABEL[c], "text": variant, "films": int(n.n),
                        "alone: R² profitability": a["log_mult"].r2, "alone: R² revenue": a["log_gross"].r2,
                        "alone: R² budget": a["log_budget"].r2, "gain, no franchise control": n0.r2_gain,
                        "R² controls only": n.r2_base, "R² gain from text": n.r2_gain, "gain 95% interval": f"{n.gain_lo:+.3f} to {n.gain_hi:+.3f}",
                        "shuffle 95th pct": n.null_p95, "shuffle p": n.p, "partial Spearman": n.spearman})
    return pd.DataFrame(out).round(3)


def checks_table(r: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in CORPORA:
        n, t = row(r, c, PRIMARY), row(r, c, "time_split")
        nr, tf = row(r, c, "net_of_rating_too"), row(r, c, PRIMARY, representation="tfidf")
        g = row(r, c, "network_assortativity", "log_mult_net_of_controls")
        out.append({"corpus": LABEL[c], "gain (5-fold)": n.r2_gain, "gain, TF-IDF instead": tf.r2_gain,
                    "TF-IDF shuffle p": tf.p, "gain, later films only": t.r2_gain, "later films": int(t.n_test),
                    "trained up to": int(t.cutoff_year), "gain beyond IMDb rating": nr.r2_gain, "rating shuffle p": nr.p,
                    "network assortativity": g.assortativity, "assortativity null 95th": g.null_p95,
                    "assortativity p": g.p})
    return pd.DataFrame(out).round(3)


def samples_table(r: pd.DataFrame) -> pd.DataFrame:
    d = r[(r.analysis == PRIMARY) & (r.representation == "embedding") & (r.variant == "clean")]
    d = d.assign(corpus=d.corpus.map(LABEL))
    return d[["corpus", "sample", "n", "r2_base", "r2_gain", "null_p95", "p", "spearman"]].rename(columns={
        "n": "films", "r2_base": "R² controls only", "r2_gain": "R² gain from text", "null_p95": "shuffle 95th pct",
        "p": "shuffle p", "spearman": "partial Spearman"}).round(3).reset_index(drop=True)


def components_table(r: pd.DataFrame) -> pd.DataFrame:
    d = pick(r, target="log_mult")
    d = d[d.analysis.str.startswith("net_k") | (d.analysis == PRIMARY)].copy()
    d["components"] = d.analysis.str.extract(r"k(\d+)")[0].fillna(50).astype(int)
    t = d.pivot(index="components", columns="corpus", values="r2_gain")[CORPORA].rename(columns=LABEL)
    return t.round(3).reset_index()


def ocr_table(r: pd.DataFrame) -> pd.DataFrame:
    """The OCR check: does damage go with profit, and the screenplay gain on every script with damage as a control."""
    d = r[r["sample"] == "every script, OCR damage as a control"]
    out = []
    for rep in ("embedding", "tfidf"):
        x = d[d.representation == rep].set_index("analysis")
        if x.empty:
            continue
        out.append({"representation": rep, "scripts": int(x.loc["ocr_vs_profit", "n"]),
                    "damaged (above cutoff)": int(x.loc["ocr_vs_profit", "damaged"]),
                    "Spearman: damage vs profit left by controls": x.loc["ocr_vs_profit", "spearman"],
                    "p": x.loc["ocr_vs_profit", "p"], "gain with damage as control": x.loc[PRIMARY, "r2_gain"],
                    "shuffle p": x.loc[PRIMARY, "p"]})
    return pd.DataFrame(out).round(3)


def words_table(w: pd.DataFrame, corpus: str) -> pd.DataFrame:
    d = w[w.corpus == corpus]
    hi = d[d.end == "more profit than expected"].sort_values("log10_ratio", ascending=False).word.tolist()
    lo = d[d.end == "less profit than expected"].sort_values("log10_ratio").word.tolist()
    return pd.DataFrame({"more profit than budget, year, genre and franchise predict": hi,
                         "less profit than they predict": lo})


# ---------------------------------------------------------------- figures

def fig_net(res: dict) -> go.Figure:
    """The main figure: what text adds to budget, year and genre, next to its shuffle null."""
    r, fig = res["results"], go.Figure()
    xs, labels = [], []
    for i, c in enumerate(CORPORA):
        for j, variant in enumerate(("raw", "clean")):
            x = i * 3 + j
            xs.append(x)
            labels.append(f"{LABEL[c]}<br>{variant} text")
            null = res["nulls"][f"{c}_{variant}_franchise"]["null"]
            fig.add_trace(go.Box(x=[x] * len(null), y=null, marker_color=NEUTRAL, line_color=NEUTRAL, boxpoints=False,
                                 width=0.5, name="shuffled within genre × decade", legendgroup="null",
                                 showlegend=(i + j == 0), hoverinfo="skip"))
            n = row(r, c, PRIMARY, variant=variant)
            fig.add_trace(go.Scatter(x=[x], y=[n.r2_gain], mode="markers", legendgroup=variant,
                                     marker=dict(color=HIGH if variant == "clean" else LOW, size=13,
                                                 symbol="circle" if variant == "clean" else "diamond"),
                                     error_y=dict(type="data", symmetric=False, array=[n.gain_hi - n.r2_gain],
                                                  arrayminus=[n.r2_gain - n.gain_lo], thickness=1.5, width=6),
                                     name=f"observed, {variant} text", showlegend=(i == 0),
                                     hovertemplate=f"{LABEL[c]}, {variant}<br>gain %{{y:+.3f}}<br>{int(n.n):,} films, "
                                                   f"shuffle p = {n.p:.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#444", line_width=1)
    fig.update_xaxes(tickvals=xs, ticktext=labels)
    fig.update_yaxes(title="Out-of-fold R² gained for ln(gross / budget)")
    return style(fig, "What the text adds once budget, year, genre and franchise are known", 470)


def fig_alone(r: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {"log_mult": HIGH, "log_gross": ACCENT, "log_budget": NEUTRAL}
    for t, name in TARGET.items():
        d = [row(r, c, "alone", t) for c in CORPORA]
        fig.add_bar(x=[LABEL[c] for c in CORPORA], y=[x.r2 for x in d], name=name.replace("<br>", ": "),
                    marker_color=colors[t],
                    customdata=[[x.pearson, x.null_p95, x.p, int(x.n)] for x in d],
                    hovertemplate="R² %{y:.3f}, r = %{customdata[0]:.2f}<br>shuffle 95th pct %{customdata[1]:.3f}, "
                                  "p = %{customdata[2]:.3f}<br>%{customdata[3]:,} films<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Out-of-fold R² from the text vectors alone")
    return style(fig, "The vectors alone: they know the budget better than the profit")


def fig_methods(r: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for rep, name, color in (("embedding", "sentence embeddings (all-MiniLM-L6-v2)", HIGH),
                             ("tfidf", "TF-IDF, 300 SVD dimensions", ACCENT)):
        d = [row(r, c, PRIMARY, representation=rep) for c in CORPORA]
        fig.add_bar(x=[LABEL[c] for c in CORPORA], y=[x.r2_gain for x in d], name=name, marker_color=color)
        fig.add_scatter(x=[LABEL[c] for c in CORPORA], y=[x.null_p95 for x in d], mode="markers",
                        name="shuffle 95th percentile", showlegend=(rep == "embedding"), offsetgroup=rep,
                        marker=dict(symbol="line-ew-open", size=26, color=LOW, line=dict(width=3)))
    fig.update_layout(barmode="group", scattermode="group")
    fig.update_yaxes(title="Out-of-fold R² gained for ln(gross / budget)")
    return style(fig, "Second method: a bag-of-words space built from the same cleaned text")


def fig_samples(r: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    d = r[(r.analysis == PRIMARY) & (r.representation == "embedding") & (r.variant == "clean")]
    colors = [HIGH, ACCENT, LOW, NEUTRAL]
    for (s, g), color in zip(d.groupby("sample", sort=False), colors):
        g = g.set_index("corpus").loc[CORPORA]
        fig.add_bar(x=[LABEL[c] for c in CORPORA], y=g.r2_gain, name=s, marker_color=color,
                    customdata=np.column_stack([g.n, g.p, g.null_p95]),
                    hovertemplate="gain %{y:+.3f}<br>%{customdata[0]:,} films, shuffle p = %{customdata[1]:.3f}"
                                  "<br>shuffle 95th pct %{customdata[2]:.3f}<extra></extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Out-of-fold R² gained for ln(gross / budget)")
    return style(fig, "The same test on other samples")


def fig_network(r: pd.DataFrame) -> go.Figure:
    names = {"log_mult": "profitability", "log_gross": "revenue",
             "log_mult_net_of_controls": "profitability net of<br>budget, year, genre, franchise"}
    fig = go.Figure()
    for c, color in zip(CORPORA, (HIGH, ACCENT, LOW)):
        d = [row(r, c, "network_assortativity", t) for t in names]
        fig.add_bar(x=list(names.values()), y=[x.assortativity for x in d], name=LABEL[c], marker_color=color,
                    customdata=[[x.null_p95, x.z, x.p] for x in d],
                    hovertemplate="assortativity %{y:.3f}<br>shuffle 95th pct %{customdata[0]:.3f}, z = %{customdata[1]:.1f}, "
                                  "p = %{customdata[2]:.3f}<extra></extra>")
        fig.add_scatter(x=list(names.values()), y=[x.null_p95 for x in d], mode="markers", offsetgroup=c,
                        name="shuffle 95th percentile", showlegend=(c == CORPORA[0]),
                        marker=dict(symbol="line-ew-open", size=22, color="#222", line=dict(width=3)))
    fig.update_layout(barmode="group", scattermode="group")
    fig.update_yaxes(title="Assortativity on the 10-nearest-neighbour network")
    return style(fig, "Text as a network: do neighbouring films have similar outcomes?")


def film_map(corpus: str, variant: str = "clean") -> pd.DataFrame | None:
    """Two principal components of the film vectors with titles and outcomes. Local only: it needs the
    vectors and the film table, and holds IMDb fields, so it is never saved."""
    f, ids = film_table(), VECTORS / f"{corpus}_ids.csv"
    if not (f.exists() and ids.exists()):
        return None
    from sklearn.decomposition import PCA
    V = np.load(VECTORS / f"{corpus}_{variant}.npy")
    d = pd.read_parquet(f).set_index("tconst").loc[pd.read_csv(ids, dtype=str).tconst].reset_index()
    P = PCA(2, random_state=0).fit_transform(V)
    return d.assign(pc1=P[:, 0], pc2=P[:, 1])


def fig_map(d: pd.DataFrame, corpus: str, color: str = "log_mult") -> go.Figure:
    names = {"log_mult": "ln(gross / budget)", "log_gross": "ln(gross)", "log_budget": "ln(budget)", "year": "year"}
    lo, hi = np.percentile(d[color], [2, 98])
    fig = go.Figure(go.Scattergl(
        x=d.pc1, y=d.pc2, mode="markers", text=d.primaryTitle + " (" + d.year.astype(str) + ")",
        customdata=np.column_stack([d.log_mult, d.genres.fillna("")]),
        marker=dict(size=5, color=d[color], cmin=lo, cmax=hi, colorscale="Viridis", colorbar=dict(title=names[color])),
        hovertemplate="%{text}<br>ln(gross / budget) %{customdata[0]:.2f}<br>%{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title="Principal component 1")
    fig.update_yaxes(title="Principal component 2")
    return style(fig, f"{LABEL[corpus]} in embedding space, coloured by {names[color]}", 560)


# ---------------------------------------------------------------- readings

def _p(p: float) -> str:
    return f"shuffle p = {p:.3f}"


def reading(res: dict) -> dict[str, str]:
    """Plain-language readings, generated from the saved numbers so the text can't drift from them."""
    r, s = res["results"], res["summary"]
    out = {}
    a = {c: {t: row(r, c, "alone", t) for t in TARGET} for c in CORPORA}
    out["alone"] = " ".join(
        f"{LABEL[c]} alone explain {a[c]['log_budget'].r2:.0%} of the variation in budget and "
        f"{a[c]['log_gross'].r2:.0%} in revenue on films the model didn't see, but only {a[c]['log_mult'].r2:.0%} "
        f"in profitability." for c in CORPORA) + (
        " The text knows how big and expensive a film is (blockbusters read like blockbusters) far better "
        "than whether it earned its budget back. Revenue is mostly budget, so a correlation with revenue "
        "alone says little about profit.")
    p = {c: row(r, c, PRIMARY) for c in CORPORA}
    n0 = {c: row(r, c, "net") for c in CORPORA}
    out["net"] = " ".join(
        f"{LABEL[c]} ({int(p[c].n):,} films) add {p[c].r2_gain:+.3f} to the R² of ln(gross / budget) "
        f"(95% interval {p[c].gain_lo:+.3f} to {p[c].gain_hi:+.3f}; shuffle 95th percentile {p[c].null_p95:+.3f}, "
        f"{_p(p[c].p)}), against {n0[c].r2_gain:+.3f} without the franchise control." for c in CORPORA)
    t = {c: row(r, c, "time_split") for c in CORPORA}
    tf = {c: row(r, c, PRIMARY, representation="tfidf") for c in CORPORA}
    rt = {c: row(r, c, "net_of_rating_too") for c in CORPORA}
    out["checks"] = " ".join(
        f"{LABEL[c]}: TF-IDF gives {tf[c].r2_gain:+.3f} ({_p(tf[c].p)}); trained on films up to "
        f"{int(t[c].cutoff_year)} and scored on the {int(t[c].n_test):,} later ones, {t[c].r2_gain:+.3f}; with the "
        f"IMDb rating also controlled, {rt[c].r2_gain:+.3f} ({_p(rt[c].p)})." for c in CORPORA)
    ag = s["score_agreement_spearman_on_shared_films"]
    out["agreement"] = (f"On the {s['shared_films_for_agreement']:,} films with all three texts, the per-film "
                        "'more profit than expected' scores of the three corpora correlate at "
                        + ", ".join(f"{k.replace('~', 'and')} ρ = {v:.2f}" for k, v in ag.items())
                        + ". The corpora partly point at the same films.")
    g = {c: row(r, c, "network_assortativity", "log_mult_net_of_controls") for c in CORPORA}
    out["network"] = " ".join(
        f"On the {LABEL[c].lower()} network ({int(g[c].edges):,} edges) the assortativity of profit left by the "
        f"controls is {g[c].assortativity:.3f} against a shuffle 95th percentile of {g[c].null_p95:.3f} "
        f"(z = {g[c].z:.1f})." for c in CORPORA)
    o = ocr_table(r)
    if not o.empty:
        e = o.iloc[0]
        out["ocr"] = (f"{int(e['damaged (above cutoff)'])} of {int(e.scripts):,} scripts have more than "
                      f"{cfg_ocr():.1%} OCR-damaged words. Damage correlates with the profit left by the controls at "
                      f"ρ = {e['Spearman: damage vs profit left by controls']:+.3f} (p = {e.p:.3f}). On every script with "
                      f"damage as a control the screenplay gain is {e['gain with damage as control']:+.3f} "
                      f"(shuffle p = {e['shuffle p']:.3f}).")
    nf = {c: row(r, c, PRIMARY, sample="no franchise films") for c in CORPORA}
    out["samples"] = " ".join(f"{LABEL[c]} without franchise films: {nf[c].r2_gain:+.3f} ({int(nf[c].n):,} films, "
                              f"{_p(nf[c].p)})." for c in CORPORA)
    return out


def cfg_ocr() -> float:
    from .paths import cfg
    return cfg()["post_hoc"]["max_ocr_damaged_word_share"]


# ---------------------------------------------------------------- leak checks

def leak_table(corpus: str) -> pd.DataFrame:
    """s04: the top 50 words at each end with the three flags; * capitalised, † fewer than 30 films."""
    a = pd.read_csv(OUTPUTS / "s04_leaks.csv")
    d = a[a.corpus == corpus]
    out = {}
    for end in ("more profit than expected", "less profit than expected"):
        e = d[d.end == end].sort_values("log10_ratio", ascending=end.startswith("less"))
        out[end] = [w + ("*" if p else "") + ("†" if f else "") for w, p, f in zip(e.word, e.proper, e.few_films)]
    n = max(map(len, out.values()))
    return pd.DataFrame({k: v + [""] * (n - len(v)) for k, v in out.items()})


def tag_entities() -> tuple[pd.DataFrame, pd.DataFrame]:
    """tag_entity_check: how the three flags do against the hand labels, and the entity tags at the extremes
    of the existing tag-profit result."""
    c = pd.read_csv(OUTPUTS / "tag_entity_check.csv", keep_default_na=False,
                    na_values={k: [""] for k in ("capitalised_share", "rho_adjusted", "rank_rho_adjusted_of_content",
                                                  "rank_rho_adjusted_votes_of_content")})
    c = c[c["class"] == "content"]
    lab = c.hand_label != ""
    flags = []
    for f in ("capitalised", "genome_name", "few_films"):
        x = c[f].astype(str).str.lower() == "true"
        flags.append({"flag": f, "tags flagged": int(x.sum()), "entities caught": f"{int((x & lab).sum())} of {int(lab.sum())}",
                      "flagged but not entities": int((x & ~lab).sum())})
    n = len(c)
    rk = c["rank_rho_adjusted_of_content"].astype(float)
    ext = c[lab & ((rk <= 25) | (rk > n - 25))][["tag", "hand_label", "rho_adjusted", "rank_rho_adjusted_of_content"]]
    return pd.DataFrame(flags), ext.sort_values("rho_adjusted", ascending=False).reset_index(drop=True)
