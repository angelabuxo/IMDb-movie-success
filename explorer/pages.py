"""The explorer's pages. Each function draws one page; app.py wires them into the navigation."""
from __future__ import annotations

import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr

from . import graph_pages as gp
from .common import (arrow_safe, DATA, FEATURES, OUTPUT_DIRS, REPO, TABULAR, flatten_json, md, render_step,
                     show_df, steps, table)

GRAPH = FEATURES / "graph"
HIGH, LOW, NEUTRAL = "#0072B2", "#D55E00", "#999999"


# ---------------------------------------------------------------- shared data

def movies() -> pd.DataFrame:
    """Movie nodes of the graph: the analysis sample with budget, gross, multiple and the clean flag."""
    n = table(GRAPH / "nodes.csv")
    m = n[n.type == "movie"].drop(columns=["birth_year", "death_year", "roles", "qid", "names"], errors="ignore").copy()
    for c in ["year", "votes", "rating", "budget_usd", "gross_usd", "multiple", "rt", "mc"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["clean"] = m["clean"].astype(str).isin(["1", "1.0", "True"])
    m["log_multiple"] = np.log(m.multiple.where(m.multiple > 0))
    m["log_budget"] = np.log(m.budget_usd.where(m.budget_usd > 0))
    m["first_genre"] = m.genres.fillna("").str.split(",").str[0].replace("", "Unknown")
    return m.rename(columns={"id": "tconst", "label": "title"})


def data_files() -> dict[str, list]:
    groups = {"data/raw/features": sorted(p for p in FEATURES.glob("*") if p.name.endswith(TABULAR)),
              "data/raw/features/graph": sorted(p for p in GRAPH.glob("*") if p.name.endswith(TABULAR))}
    for k, d in OUTPUT_DIRS.items():
        groups[f"{d.relative_to(REPO)}"] = sorted(p for p in d.glob("*") if p.name.endswith(TABULAR))
    return {k: v for k, v in groups.items() if v}


# ---------------------------------------------------------------- home

def home():
    st.title("What makes a movie highly profitable?")
    st.markdown("A local explorer for the DTU 02805 final project. It shows what the notebooks show, from the same "
                "saved outputs, lets you play with the data, and lets you dig into every file to check it.")
    have_data = (GRAPH / "nodes.csv").exists()
    c = st.columns(4)
    c[0].metric("Data folder", "found" if have_data else "missing")
    if have_data:
        m = movies()
        c[1].metric("Movies in the graph", f"{len(m):,}")
        c[2].metric("Clean (budget and gross usable)", f"{int(m.clean.sum()):,}")
        c[3].metric("Clean, 100k+ votes", f"{int((m.clean & (m.tier == '100k+')).sum()):,}")
    else:
        st.warning(f"No data at `{DATA}`. Run `python scripts/get_data.py` or unpack `data/bundles/` first "
                   "(see the README). The notebook pages still work from the committed outputs.")
    st.subheader("Pages")
    st.markdown("""
- **Screenplay arcs** and **Screenplay text**: every step of the two notebook series, rendered from their saved
  outputs. Tables are sortable and filterable, figures are interactive.
- **Network**: the four network analyses (return similarity, pair returns, team history, movers), the per-film
  network features, and a lookup for any film's crew and the films it is linked to.
- **Profit playground**: choose the sample, axes and colours, and explore budget, gross and return.
- **Data inspector**: open any data or output file and check its columns, missing values, duplicates and rows.
- **Checks**: automatic sanity checks on the data and outputs, and the test suite.
""")
    st.caption(f"Data folder: `{DATA}`. Nothing here writes to the repo or the data folder.")


# ---------------------------------------------------------------- notebook series

def _series(pkg: str, title: str, intro: str):
    st.title(title)
    st.caption(intro)
    opts = steps(pkg)
    labels = [f"{n} · {slug.replace('_', ' ')}" for n, _, slug in opts]
    with st.sidebar:
        pick = st.radio("Step", labels, key=f"step_{pkg}")
        show_private = st.toggle("Show private text (local only)", key=f"priv_{pkg}",
                                 help="Script and review excerpts. Only from files already on this machine; never published.")
    num, module_name, slug = opts[labels.index(pick)]
    render_step(pkg, module_name, show_private)


def arcs():
    _series("arcs", "Screenplay arcs", "notebooks/arcs/*: sentiment arcs of screenplays against log(gross/budget). "
            "Same saved outputs as the notebooks; nothing is recomputed.")


def text():
    _series("scripts_text", "Screenplay text", "notebooks/scripts_text/*: which words separate profitable from "
            "unprofitable scripts. Same saved outputs as the notebooks; nothing is recomputed.")


# ---------------------------------------------------------------- network

def _null_chart(df: pd.DataFrame, x: str, obs: str, null95: str, title: str):
    fig = go.Figure()
    fig.add_bar(x=df[x].astype(str), y=df[obs], name="observed", marker_color=HIGH)
    fig.add_scatter(x=df[x].astype(str), y=df[null95], name="null 95th percentile", mode="markers",
                    marker=dict(symbol="line-ew-open", size=28, color=LOW, line=dict(width=3)))
    fig.update_layout(template="simple_white", title=title, height=380, legend=dict(orientation="h", y=1.1))
    return fig


def network():
    st.title("Network")
    st.caption("scripts/network/*.py and the notebooks movie_graph, network_reach and what_predicts_profit.")
    t1, t2, t3, t4 = st.tabs(["Results", "Film features", "Film lookup", "Graph"])

    with t1:
        name = st.selectbox("Result file", ["roi_similarity", "pair_returns", "team_history", "movers"])
        res = json.loads((REPO / "analysis" / "network" / f"{name}.json").read_text())
        if name == "pair_returns":
            variant = st.radio("Sample", list(res["graph"]), horizontal=True)
            d = pd.DataFrame(res["graph"][variant]["distance"]).T.rename_axis("distance").reset_index()
            st.plotly_chart(_null_chart(d, "distance", "corr", "null_all_95",
                                        "Return correlation of film pairs by distance in the people graph"), width="stretch")
            k = pd.DataFrame(res["graph"][variant]["dose"]).T.rename_axis("shared people").reset_index()
            st.plotly_chart(_null_chart(k, "shared people", "corr", "null_all_95",
                                        "Directly linked pairs, by number of people shared"), width="stretch")
        elif name == "roi_similarity":
            rows = [{"variant": v, "role": r, **s} for v, x in res["results"].items() for r, s in x["roles"].items()]
            d = pd.DataFrame(rows)
            variant = st.selectbox("Variant", d.variant.unique().tolist(), index=1)
            dv = d[d.variant == variant]
            st.plotly_chart(_null_chart(dv, "role", "similarity", "similarity_null_95",
                                        f"How alike are returns of films sharing a {('person or company')}? ({variant})"),
                            width="stretch")
            st.markdown("**All variants side by side** (similarity excess over the shuffle null):")
            st.dataframe(d.pivot(index="role", columns="variant", values="similarity_excess").round(3), width="stretch")
        elif name == "team_history":
            variant = st.radio("Sample", list(res["results"]), horizontal=True)
            d = pd.DataFrame(res["results"][variant]["measures"]).T.rename_axis("measure").reset_index()
            st.plotly_chart(_null_chart(d, "measure", "rho", "rho_null_95", "Team history vs return (Spearman rho)"),
                            width="stretch")
        elif name == "movers":
            variant = st.radio("Sample", list(res["results"]), horizontal=True)
            rows = [{"role": r, "person": s["movers"]["person"], "person_null_95": s["movers"]["person_null_95"],
                     "studio": s["movers"]["studio"], "studio_null_95": s["movers"]["studio_null_95"]}
                    for r, s in res["results"][variant].items() if "movers" in s]
            d = pd.DataFrame(rows)
            fig = go.Figure()
            fig.add_bar(x=d.role, y=d.person, name="person's past record", marker_color=HIGH)
            fig.add_bar(x=d.role, y=d.studio, name="studio's past record", marker_color=LOW)
            fig.update_layout(template="simple_white", barmode="group", height=380,
                              title="When people move studio, whose track record predicts the new film?")
            st.plotly_chart(fig, width="stretch")
        with st.expander("Every number in this file", expanded=False):
            show_df(flatten_json(res), key=f"json_{name}")
        with st.expander("Raw JSON"):
            st.json(res, expanded=1)

    if not (GRAPH / "nodes.csv").exists():
        for t in (t2, t3, t4):
            t.warning("Needs the data folder (data/raw/features/graph).")
        return
    m = movies()

    with t2:
        f = table(FEATURES / "film_network_features.csv")
        d = f.merge(m[["tconst", "title", "clean", "tier", "log_multiple", "first_genre"]], on="tconst")
        d = d[d.clean & d.log_multiple.notna()]
        feats = [c for c in f.columns if c not in ("tconst", "year") and pd.api.types.is_numeric_dtype(f[c])]
        c1, c2, c3 = st.columns(3)
        feat = c1.selectbox("Feature (built from earlier films only)", feats, index=feats.index("cast_track") if "cast_track" in feats else 0)
        tier = c2.selectbox("Tier", ["all", "100k+", "10k-100k"])
        yr = c3.slider("Release year", 1950, 2025, (1980, 2025))
        if tier != "all":
            d = d[d.tier == tier]
        d = d[d.year.between(*yr)]
        x = d[[feat, "log_multiple"]].dropna()
        if len(x) > 10:
            rho = spearmanr(x[feat], x.log_multiple).statistic
            st.metric(f"Spearman rho with log(gross/budget), {len(x):,} films", f"{rho:+.3f}")
            fig = px.scatter(d, x=feat, y="log_multiple", hover_name="title", hover_data=["year", "first_genre"],
                             opacity=.35, template="simple_white", height=480, color_discrete_sequence=[HIGH])
            q = x.assign(bin=pd.qcut(x[feat].rank(method="first"), 10, labels=False)).groupby("bin").mean()
            fig.add_scatter(x=q[feat], y=q.log_multiple, mode="lines+markers", name="decile means",
                            line=dict(color=LOW, width=3))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Fewer than 10 films with this feature in the selection.")
        show_df(d[["tconst", "title", "year", "tier", "log_multiple", feat]].sort_values(feat, ascending=False),
                key="netfeat", caption="Films in the selection, highest feature value first")

    with t3:
        e = table(GRAPH / "edges.csv")
        nodes = table(GRAPH / "nodes.csv")[["id", "type", "label"]]
        q = st.text_input("Film title or IMDb ID", "Inception")
        hits = m[m.title.str.contains(q, case=False, na=False, regex=False) | (m.tconst == q)].sort_values("votes", ascending=False)
        if hits.empty:
            st.info("No film matches.")
        else:
            pick = st.selectbox("Film", hits.tconst, format_func=lambda t: f"{hits.set_index('tconst').title[t]} ({int(hits.set_index('tconst').year[t])})")
            row = m.set_index("tconst").loc[pick]
            c = st.columns(5)
            c[0].metric("Budget", f"${row.budget_usd / 1e6:,.1f}M" if pd.notna(row.budget_usd) else "—")
            c[1].metric("Gross", f"${row.gross_usd / 1e6:,.1f}M" if pd.notna(row.gross_usd) else "—")
            c[2].metric("Gross ÷ budget", f"{row.multiple:,.2f}" if pd.notna(row.multiple) else "—")
            c[3].metric("Clean", "yes" if row.clean else "no")
            c[4].metric("IMDb", f"{row.rating} · {int(row.votes):,} votes" if pd.notna(row.votes) else "—")
            crew = e[e.target == pick].merge(nodes, left_on="source", right_on="id", how="left")
            st.markdown("**Credits in the graph**")
            show_df(crew[["role", "ordering", "label", "source", "origin"]].sort_values(["role", "ordering"]), key="crew")
            people = crew[crew.type == "person"].source
            linked = e[e.source.isin(people) & (e.target != pick)]
            shared = (linked.groupby("target").source.nunique().rename("shared_people").reset_index()
                      .merge(m[["tconst", "title", "year", "multiple", "clean"]], left_on="target", right_on="tconst")
                      .sort_values(["shared_people", "year"], ascending=[False, True]))
            st.markdown(f"**Films linked through shared people** ({len(shared):,}). Earlier films are what the "
                        "leakage-safe features may use.")
            show_df(shared.drop(columns="target").assign(earlier=shared.year < row.year), key="linked")

    with t4:
        n = table(GRAPH / "nodes.csv")
        e = table(GRAPH / "edges.csv")
        c = st.columns(3)
        c[0].metric("Nodes", f"{len(n):,}")
        c[1].metric("Edges (credits)", f"{len(e):,}")
        c[2].metric("Wikipedia-only producer credits", f"{int((e.origin != 'imdb').sum()):,}" if "origin" in e else "—")
        st.plotly_chart(px.bar(n.type.value_counts().reset_index(), x="type", y="count", template="simple_white",
                               title="Nodes by type", height=320), width="stretch")
        st.plotly_chart(px.bar(e.groupby(["role", "origin"]).size().rename("credits").reset_index(), x="role",
                               y="credits", color="origin", template="simple_white", title="Credits by role and source",
                               height=360), width="stretch")


# ---------------------------------------------------------------- profit playground

def profit():
    st.title("Profit playground")
    if not (GRAPH / "nodes.csv").exists():
        st.warning("Needs the data folder (data/raw/features/graph/nodes.csv).")
        return
    m = movies()
    with st.sidebar:
        st.subheader("Sample")
        clean = st.toggle("Clean films only", True, help="Usable USD budget and worldwide gross, released 1975 to 2025, no streaming, rentals or direct-to-video.")
        tiers = st.multiselect("Vote tier", sorted(m.tier.dropna().unique()), default=sorted(m.tier.dropna().unique()))
        yr = st.slider("Release year", int(m.year.min()), int(m.year.max()), (1980, int(m.year.max())))
        genres = sorted({g for gs in m.genres.dropna() for g in gs.split(",") if g})
        need = st.multiselect("Must include genre", genres)
        bud = st.slider("Budget (USD millions)", 0.0, 400.0, (0.0, 400.0), step=1.0)
    d = m[m.tier.isin(tiers) & m.year.between(*yr)]
    if clean:
        d = d[d.clean]
    for g in need:
        d = d[d.genres.fillna("").str.split(",").apply(lambda s, g=g: g in s)]
    d = d[(d.budget_usd.fillna(0) / 1e6).between(*bud)]
    st.caption(f"{len(d):,} films in the selection")
    if d.empty:
        return
    c = st.columns(4)
    c[0].metric("Median gross ÷ budget", f"{d.multiple.median():.2f}")
    c[1].metric("Grossed less than budget", f"{(d.multiple < 1).mean():.1%}")
    c[2].metric("Median budget", f"${d.budget_usd.median() / 1e6:,.1f}M")
    c[3].metric("Median IMDb rating", f"{d.rating.median():.1f}")

    arcs = gp.arc_tables()
    k_arc, names = None, pd.Series(dtype=str)
    if arcs:
        with st.sidebar:
            st.subheader("Sentiment arcs")
            k_arc = st.radio("Arc instrument", list(arcs), horizontal=True, key="profit_arc_inst",
                             format_func=lambda x: gp.INSTRUMENTS[x] + (" (primary)" if x == gp.PRIMARY else ""))
        names = gp.script_names()
        desc = arcs[k_arc]["desc"]
        mn = d.tconst.map(names)
        d = d.assign(**{col: mn.map(desc[src]) for col, (src, _) in ARC_COLS.items()},
                     arc_in_primary=mn.map(desc.in_primary.astype(bool)).fillna(False).astype(bool))
    num = [c for c in ["log_budget", "log_multiple", "budget_usd", "gross_usd", "multiple", "year", "votes", "rating", "rt", "mc"]
           + list(ARC_COLS) if c in d and d[c].notna().any()]
    a, b, col, sz = st.columns(4)
    x = a.selectbox("x", num, index=num.index("log_budget"))
    ys = [c for c in num if c != x]  # same column on both axes breaks the trendline
    y = b.selectbox("y", ys, index=ys.index("log_multiple") if "log_multiple" in ys else 0)
    colour = col.selectbox("colour", ["first_genre", "tier", "clean", "none"])
    trend = sz.toggle("Trend line", True)
    fig = px.scatter(d, x=x, y=y, color=None if colour == "none" else colour, hover_name="title",
                     hover_data=["tconst", "year", "multiple", "rating"], opacity=.5, template="simple_white", height=560,
                     trendline="lowess" if trend else None, trendline_scope="overall" if trend else "trace",
                     trendline_color_override=LOW if trend else None)
    st.plotly_chart(fig, width="stretch")
    xy = d[[x, y]].dropna()
    if len(xy) > 5:
        st.caption(f"Spearman rho = {spearmanr(xy[x], xy[y]).statistic:+.3f} over {len(xy):,} films. "
                   "Gross over budget is not profit: budgets leave out marketing, and cinemas keep part of the gross.")

    if k_arc:
        _profit_arcs(d, arcs[k_arc], k_arc, names)

    l, r = st.columns(2)
    l.plotly_chart(px.histogram(d, x="log_multiple", nbins=60, template="simple_white", height=360,
                                title="log(gross / budget): 0 means it grossed its budget",
                                color_discrete_sequence=[HIGH]).add_vline(x=0, line_dash="dash"), width="stretch")
    g = d.groupby("first_genre").log_multiple.median().sort_values().index.tolist()
    r.plotly_chart(px.box(d, x="first_genre", y="log_multiple", category_orders={"first_genre": g}, template="simple_white",
                          height=360, title="By first-listed genre", color_discrete_sequence=[NEUTRAL]), width="stretch")
    st.subheader("Biggest hits and flops in the selection")
    k = st.slider("How many", 5, 100, 20)
    cols = ["tconst", "title", "year", "genres", "budget_usd", "gross_usd", "multiple", "rating", "votes", "tier"]
    h, f = st.columns(2)
    with h:
        show_df(d.nlargest(k, "multiple")[cols], key="hits", caption="Highest gross ÷ budget")
    with f:
        show_df(d.nsmallest(k, "multiple")[cols], key="flops", caption="Lowest gross ÷ budget")


ARC_COLS = {"arc_swing": ("D1", "swing (D1)"), "arc_ending_lift": ("D2", "ending lift (D2)"),
            "arc_mean_level": ("mean_level", "mean level"), "arc_low_point": ("min_position", "low point (share of story)"),
            "arc_swings": ("sign_changes", "swings")}
SPLITS = {"bottom vs top third of log(gross / budget)": 1 / 3, "bottom vs top quarter": 1 / 4,
          "grossed less vs more than its budget": None}


def _profit_arcs(d: pd.DataFrame, arcs: dict, k: str, names: pd.Series):
    """Mean sentiment arc of the least and most profitable films in the selection, and how each arc
    descriptor correlates with return there."""
    st.subheader("Sentiment arcs and return")
    a = d[d.arc_swing.notna() & d.log_multiple.notna()]
    if st.toggle("Primary arcs sample only", True, key="profit_arc_primary",
                 help="The films the arcs analysis and the paper use. Off adds scripts it set aside, such as possible "
                      "transcripts or scripts with too few scene tags."):
        a = a[a.arc_in_primary]
    st.caption(f"{len(a):,} of the {len(d):,} films in the selection have a saved arc ({gp.INSTRUMENTS[k]}). "
               "Exploratory: the arcs notebook's Step 7 is the test fixed in advance.")
    if len(a) < 20:
        st.info("Fewer than 20 films with an arc in this selection. Widen the filters.")
        return
    split = st.radio("Compare", list(SPLITS), horizontal=True, key="profit_arc_split")
    q = SPLITS[split]
    if q is None:
        lo, hi = a[a.multiple < 1], a[a.multiple >= 1]
        names_ = ("grossed less than budget", "grossed more")
    else:
        lo, hi = a[a.log_multiple <= a.log_multiple.quantile(q)], a[a.log_multiple >= a.log_multiple.quantile(1 - q)]
        names_ = ("least profitable", "most profitable")
    curves = arcs["arcs"]
    fig = go.Figure()
    for grp, colour, label, fill in ((lo, LOW, names_[0], "rgba(213,94,0,.15)"), (hi, HIGH, names_[1], "rgba(0,114,178,.15)")):
        if grp.empty:
            continue
        films = curves.index.intersection(grp.tconst.map(names).dropna().unique())
        w = curves.loc[films].reset_index().pivot_table(index="t", columns="movie_name", values="smooth")
        m, se = w.mean(axis=1), w.std(axis=1) / np.sqrt(w.shape[1])
        fig.add_scatter(x=np.r_[m.index, m.index[::-1]], y=np.r_[m + 1.96 * se, (m - 1.96 * se)[::-1]], fill="toself",
                        fillcolor=fill, line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=m.index, y=m.values, mode="lines", line=dict(color=colour, width=3),
                        name=f"{label} (n={w.shape[1]:,})", hovertemplate="story %{x}%: %{y:+.3f}<extra></extra>")
    fig.add_hline(y=0, line_dash="dot", line_color="#aaa")
    fig.update_layout(template="simple_white", height=420, xaxis_title="story time (%)",
                      yaxis_title=f"mean smoothed arc ({gp.INSTRUMENTS[k]})", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch", key="profit_arcs")
    st.caption("Lines: the mean of the films' smoothed arcs. Bands: ±1.96 standard errors across films.")
    rows = []
    for col, (_, label) in ARC_COLS.items():
        x = a[[col, "log_multiple"]].dropna()
        if len(x) > 5 and x[col].nunique() > 1:   # a constant descriptor has no correlation
            r = spearmanr(x[col], x.log_multiple)
            rows.append({"descriptor": label, "Spearman rho with log(gross / budget)": round(r.statistic, 3),
                         "p": float(f"{r.pvalue:.2g}"), "films": len(x)})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------- data inspector

MAX_COLS = 500


def inspector():
    st.title("Data inspector")
    groups = data_files()
    if not groups:
        st.warning("No data or output files found.")
        return
    with st.sidebar:
        g = st.selectbox("Folder", list(groups))
        f = st.selectbox("File", groups[g], format_func=lambda p: f"{p.name}  ({p.stat().st_size / 1e6:,.1f} MB)")
        limit = st.toggle("Read only the first 200,000 rows", f.stat().st_size > 150e6)
    try:
        df = table(f, 200_000 if limit else None)
    except Exception as e:  # noqa: BLE001
        st.error(f"Can't read `{f.name}`: {e}")
        return
    st.subheader(f.name)
    st.caption(f"`{f}` · {len(df):,} rows × {df.shape[1]:,} columns" + (" (first 200,000 only)" if limit else ""))
    if df.shape[1] > MAX_COLS:   # e.g. screenplay_words.json flattens to 160,000+ word columns, too big to send
        st.warning(f"{df.shape[1]:,} columns is more than the browser can be sent; showing the first {MAX_COLS:,}.")
        df = df.iloc[:, :MAX_COLS]

    schema = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values,
                           "non-null": df.notna().sum().values, "missing %": (df.isna().mean() * 100).round(1).values,
                           "unique": [df[c].nunique(dropna=True) for c in df.columns],
                           "example": [str(df[c].dropna().iloc[0])[:80] if df[c].notna().any() else "" for c in df.columns]})
    t1, t2, t3, t4 = st.tabs(["Rows", "Columns", "One column", "Duplicates"])
    with t1:
        show_df(df, key=f"rows_{f.name}", height=520)
    with t2:
        st.dataframe(arrow_safe(schema), width="stretch", hide_index=True)
        numeric = df.select_dtypes("number")
        if not numeric.empty:
            st.markdown("**Numeric summary**")
            st.dataframe(numeric.describe(percentiles=[.01, .25, .5, .75, .99]).T, width="stretch")
    with t3:
        col = st.selectbox("Column", df.columns)
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            log = st.toggle("Log scale (positive values)", False)
            v = np.log10(s[s > 0]) if log else s
            st.plotly_chart(px.histogram(v.dropna(), nbins=80, template="simple_white", height=380,
                                         title=f"{'log10 ' if log else ''}{col}"), width="stretch")
            st.dataframe(s.describe().to_frame().T, width="stretch")
        vc = s.astype(str).value_counts(dropna=False).head(50).rename_axis("value").reset_index()
        st.markdown("**Most common values**")
        st.dataframe(vc, width="stretch", hide_index=True)
    with t4:
        unique_by = {"nodes.csv": ["id"], "people.csv": ["nconst"], "movies_features.csv": ["tconst"],
                     "movies_features_10k.csv": ["tconst"], "film_network_features.csv": ["tconst"],
                     "wiki_infobox.csv": ["tconst"], "wiki_infobox_10k.csv": ["tconst"], "edges.csv": ["source", "target", "role"]}
        default = [c for c in unique_by.get(f.name, []) if c in df.columns]
        keys = st.multiselect("Key columns (should be unique)", df.columns.tolist(), default=default,
                              help="Pre-filled only for files where the key is meant to be unique.")
        if keys:
            dup = df[df.duplicated(keys, keep=False)].sort_values(keys)
            (st.success if dup.empty else st.error)(f"{len(dup):,} rows share a key with another row.")
            if not dup.empty:
                show_df(dup, key=f"dup_{f.name}")


# ---------------------------------------------------------------- checks

CAMEL = re.compile(r"(?<=[a-z])(?=[A-Z][a-z])")   # a point where two names could have been glued together

# Findings a person has checked. A check lists them but doesn't count them against the data.
VERIFIED_STALE = {"s03_compute_check": "a timing benchmark that reads only seed, instruments.compute_check_sentences, "
                  "instruments.contextual and instruments.max_runtime_hours, none of which changed after it ran "
                  "(checked 2026-11-15); rerunning needs the private sentence table"}
VERIFIED_ODD = {"tt13570066": "Desert Warrior (2025): $150M budget and $759,228 gross are both what the cited sources "
                "report (checked 2026-11-15); a real flop"}


def merged_names(names: pd.Series) -> pd.Series:
    """Names that are two other names glued together ('Madras TalkiesLyca Productions'): split at a lower-to-upper
    boundary, both halves appear on their own elsewhere. Real CamelCase names ('DreamWorks Pictures') don't."""
    known = set(names)
    def glued(x):
        return any(x[:m.start()].strip() in known and x[m.start():].strip() in known for m in CAMEL.finditer(x))
    return names.map(glued)


def readme_counts() -> dict:
    """The sample sizes the README states, so the check follows the README instead of fixed numbers."""
    t = (REPO / "README.md").read_text()
    def num(pat):
        m = re.search(pat, t)
        return int(m.group(1).replace(",", "")) if m else None
    return {"clean": num(r"It uses the ([\d,]+) clean movies"), "clean_100k": num(r"([\d,]+) movies are left")}


def _check(name: str, ok: bool, detail: str, review: bool = False):
    return {"check": name, "status": "pass" if ok else ("review" if review else "FAIL"), "detail": detail}


def checks():
    st.title("Checks")
    st.caption("Automatic sanity checks on the data and saved outputs. **pass**: the property holds. **FAIL**: it "
               "doesn't. **review**: a list worth a human look, shown below the table.")
    rows, lists = [], {}
    for pkg in ("arcs", "scripts_text"):
        cfgmod = __import__(f"analysis.{pkg}.config", fromlist=["config_hash", "RUN_LOG"])
        log = json.loads(cfgmod.RUN_LOG.read_text()) if cfgmod.RUN_LOG.exists() else {}
        now = cfgmod.config_hash()
        older = sorted(k for k, v in log.items() if v.get("config") != now)
        stale = [k for k in older if k not in VERIFIED_STALE]
        rows.append(_check(f"{pkg}: every saved step was run with the current config.yaml (or verified unaffected)",
                           not stale, f"current hash {now}; older and unverified: {', '.join(stale) or 'none'}; "
                           f"older but verified: {', '.join(k for k in older if k in VERIFIED_STALE) or 'none'}", review=True))
        if older:
            lists[f"{pkg}: steps saved under an older config.yaml"] = pd.DataFrame(
                [{"step": k, "saved": log[k].get("date"), "config hash": log[k].get("config"),
                  "verified": VERIFIED_STALE.get(k, "no: rerun it, or check the changed keys can't affect it")} for k in older])
    if (GRAPH / "nodes.csv").exists():
        m = movies()
        rows.append(_check("movies: one row per IMDb ID", m.tconst.is_unique, f"{len(m):,} movies"))
        c = m[m.clean]
        rows.append(_check("clean movies have positive budget and gross",
                           bool((c.budget_usd > 0).all() and (c.gross_usd > 0).all()), f"{len(c):,} clean movies"))
        gap = (c.multiple - c.gross_usd / c.budget_usd).abs()
        tol = 5e-5 + 1e-3 * c.multiple                     # the multiple is saved to 4 decimals
        rows.append(_check("multiple equals gross ÷ budget (up to its 4-decimal rounding)", bool((gap <= tol).all()),
                           f"{int((gap > tol).sum())} films outside the tolerance; largest gap {gap.max():.1e}"))
        odd = c[(c.budget_usd >= 1e8) & (c.multiple < 0.01)]
        new_odd = odd[~odd.tconst.isin(VERIFIED_ODD)]
        rows.append(_check("no unverified clean film with a $100M+ budget and under 1% of it back (likely a currency error)",
                           new_odd.empty, f"{len(new_odd)} unverified ({', '.join(new_odd.title.head(3)) or 'none'}); "
                           f"{len(odd) - len(new_odd)} verified as real", review=True))
        if len(odd):
            lists["Clean films with a $100M+ budget and under 1% of it back"] = \
                odd[["tconst", "title", "year", "budget_usd", "gross_usd", "multiple"]].assign(
                    verified=odd.tconst.map(VERIFIED_ODD).fillna("no: check the infobox currency"))
        ext = c[(c.multiple < 0.005) | (c.multiple > 200)].sort_values("multiple")
        lists[f"Extreme multiples among clean films ({len(ext)}): below 0.005 or above 200, mostly real but worth a look"] = \
            ext[["tconst", "title", "year", "budget_usd", "gross_usd", "multiple"]]
        rows.append(_check("release years are plausible (1900 to 2026)", bool(m.year.dropna().between(1900, 2026).all()),
                           f"{int(m.year.min())} to {int(m.year.max())}"))
        rc = readme_counts()
        rows.append(_check("README's clean sample size matches the data", len(c) == rc["clean"],
                           f"README {rc['clean']:,}, data {len(c):,}" if rc["clean"] else "README number not found"))
        n100 = int((c.tier == "100k+").sum())
        rows.append(_check("README's 100k+ clean count matches the data", n100 == rc["clean_100k"],
                           f"README {rc['clean_100k']:,}, data {n100:,}" if rc["clean_100k"] else "README number not found"))
        if (FEATURES / "film_network_features.csv").exists():
            f = table(FEATURES / "film_network_features.csv")
            rows.append(_check("film features: one row per movie", f.tconst.is_unique and set(f.tconst) == set(m.tconst),
                               f"{len(f):,} rows for {len(m):,} movies"))
            bad = f[(f.director_track_n.fillna(0) < 2) & f.director_track.notna()]
            rows.append(_check("film features: no track record from fewer than two earlier films", bad.empty, f"{len(bad):,} violations"))
    for name in ("wiki_infobox.csv", "wiki_infobox_10k.csv"):
        p = FEATURES / name
        if p.exists():
            ib = table(p)
            vals = pd.concat([ib[["tconst", "title", col]].rename(columns={col: "name"}).assign(field=col)
                              for col in ("studios", "distributors", "producers", "starring")])
            vals = vals.assign(name=vals["name"].fillna("").astype(str).str.split("|")).explode("name")
            vals = vals[vals["name"] != ""]
            hit = vals[merged_names(vals["name"])]
            rows.append(_check(f"{name}: no names glued together like 'Madras TalkiesLyca Productions'", hit.empty,
                               f"{len(hit)} of {len(vals):,} names split into two names found elsewhere"))
            if len(hit):
                lists[f"{name}: names that look glued together"] = hit[["tconst", "title", "field", "name"]]
    t = pd.DataFrame(rows)
    colour = {"pass": "color:#0a7d32;font-weight:600", "FAIL": "color:#b00020;font-weight:700",
              "review": "color:#a15c00;font-weight:700"}
    st.dataframe(t.style.map(lambda v: colour.get(v, ""), subset=["status"]), width="stretch", hide_index=True)
    for title, df in lists.items():
        with st.expander(f"{title} ({len(df):,})"):
            show_df(df, key="chk_" + str(abs(hash(title))))

    st.subheader("Test suite")
    st.caption("tests/: the functions behind every notebook and every explorer page, on synthetic data only.")
    fast = st.toggle("Skip the slow test", True)
    if st.button("Run tests"):
        with st.spinner("Running pytest…"):
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + (["-m", "not slow"] if fast else []),
                               cwd=REPO, capture_output=True, text=True)
        (st.success if r.returncode == 0 else st.error)(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "no output")
        with st.expander("Full output"):
            st.code(r.stdout[-20000:] + r.stderr[-5000:])
    if (REPO / "TESTING.md").exists():
        with st.expander("TESTING.md (the testing audit)"):
            st.markdown(md((REPO / "TESTING.md").read_text()))


# ---------------------------------------------------------------- TMDB as a second money source

TMDB_JSON = REPO / "analysis" / "checks" / "tmdb_compare.json"
TMDB_CSV = FEATURES / "tmdb_compare.csv"
TMDB_FIELDS = {"budget": ("wiki_budget", "tmdb_budget", "budget_ratio"), "gross": ("wiki_gross", "tmdb_revenue", "gross_ratio")}


def tmdb():
    st.title("Second money source: TMDB")
    st.caption("Budget and gross are parsed from Wikipedia infoboxes. This page checks them against TMDB's budget "
               "and revenue for the same films, matched on IMDb ID. Agreement isn't proof: TMDB is edited by users "
               "without citations, and both sources can copy the same wrong number.")
    if not TMDB_JSON.exists():
        st.warning("No saved comparison yet. With the TMDB file in `data/raw/tmdb/`, run "
                   "`../../.venv/bin/python ../../scripts/imdb/tmdb_compare.py` from `data/raw`.")
        return
    res = json.loads(TMDB_JSON.read_text())
    tier = st.sidebar.radio("Vote tier", list(res["tiers"]), key="tmdb_tier")
    field = st.sidebar.radio("Money field", list(TMDB_FIELDS), key="tmdb_field")
    s = res["tiers"][tier]
    f = s[field]
    st.caption(f"TMDB snapshot of {res['tmdb_snapshot']}. {res['clean_movies']:,} clean movies, "
               f"{res['matched_to_tmdb_row']:,} with a TMDB row.")
    c = st.columns(4)
    c[0].metric(f"TMDB has a {field} figure", f"{f['n']:,}", f"{f['tmdb_missing']:,} without", delta_color="off")
    c[1].metric("Within 10% of Wikipedia", f"{f['within_10pct']:.1%}")
    c[2].metric("Off by 2x or more", f"{f['off_by_2x_or_more']:.1%}")
    c[3].metric("Same under-budget call", f"{s['same_under_budget_call']:.1%}",
                f"on {s['both_tmdb_fields']:,} films with both TMDB figures", delta_color="off")

    rows = [{"tier": t, "field": k, "clean movies": v["n"], "TMDB has a figure": v[k]["n"],
             "within 1%": v[k]["within_1pct"], "within 10%": v[k]["within_10pct"], "within 25%": v[k]["within_25pct"],
             "exactly equal": v[k].get("exactly_equal"), "within 0.5%": v[k].get("identical_within_half_pct"),
             "off by 2x or more": v[k]["off_by_2x_or_more"], "TMDB 2x higher": v[k]["tmdb_higher_2x"],
             "TMDB at most half": v[k]["tmdb_lower_2x"], "Spearman (logs)": v[k]["spearman_log"]}
            for t, v in res["tiers"].items() for k in TMDB_FIELDS]
    show_df(pd.DataFrame(rows), key="tmdb_agreement", caption="Agreement by tier and field (shares are of the movies "
            "where TMDB has a figure).")

    if "exactly_equal" in f:
        st.caption(f"The sources aren't independent: {f['exactly_equal']:.1%} of TMDB's {field} figures in this tier "
                   f"are exactly Wikipedia's number and {f['identical_within_half_pct']:.1%} are within 0.5%. Read the "
                   "agreement as a check on the parser, not on Wikipedia.")
    if "flips" in s:
        st.subheader("Films that change their flop label")
        show_df(pd.DataFrame([{"tier": t, "films with both TMDB figures": v["both_tmdb_fields"], **v["flips"]}
                              for t, v in res["tiers"].items()]), key="tmdb_flips",
                caption=f"A flip near break-even (Wikipedia gross ÷ budget within a factor of "
                        f"{res.get('near_break_even_factor', 1.25)} of 1) is expected. One elsewhere is a large disagreement.")
    if res.get("outside_clean"):
        st.subheader("Films the clean rule drops")
        show_df(pd.DataFrame(res["outside_clean"]), key="tmdb_outside",
                caption=f"{res['all_movies'] - res['clean_movies']:,} of {res['all_movies']:,} films aren't clean. Everything "
                        "above only tests clean films, and a parser miss usually drops a film from the sample. Films "
                        "with no parsed figure but both figures on TMDB are the ones to read next.")

    if res.get("unparsed_with_tmdb"):
        show_df(pd.DataFrame(res["unparsed_with_tmdb"]), key="tmdb_unparsed",
                caption="Of the films with no parsed Wikipedia figure but both on TMDB: where the figure went missing. "
                        "Only the last row can hold parser misses.")
    rob = REPO / "analysis" / "checks" / "tmdb_robustness.json"
    if rob.exists():
        rb = json.loads(rob.read_text())
        a, sr = rb["arcs_step7"], rb["scripts_and_reviews"]
        runs = {"reproduce": "committed (Wikipedia)", "wikipedia": "TMDB films, Wikipedia money", "tmdb": "TMDB films, TMDB money"}
        rows = [{"result": f"arcs {k}", **{lab: f"{a[k][v]['rho']:+.3f} (p {a[k][v]['p']:.3f})" for v, lab in runs.items()}}
                for k in ("D1", "D2")]
        rows += [{"result": f"script: {f}", **{lab: f"{sr[v]['features'][f]['rho']:+.3f} (q {sr[v]['features'][f]['q_partial']:.3f})"
                                               for v, lab in runs.items()}}
                 for f in ("degree centralisation", "network density", "clustering")]
        rows.append({"result": "review sentiment", **{lab: f"{sr[v]['review_sentiment']['rho']:+.3f}" for v, lab in runs.items()}})
        st.subheader("Do the results change with TMDB's money?")
        show_df(pd.DataFrame(rows), key="tmdb_robustness",
                caption="Each analysis's own code: the committed number first, then the films where TMDB has both "
                        "figures, with Wikipedia's money and with TMDB's (outcome, budget control and hit/flop cut-offs "
                        "all swapped). The network results aren't covered.")

    edges = np.array(res["ratio_hist_edges_log10"])
    h = pd.DataFrame({"log10(TMDB / Wikipedia)": (edges[:-1] + edges[1:]) / 2, "movies": f["ratio_hist"]})
    fig = px.bar(h[h.movies > 0], x="log10(TMDB / Wikipedia)", y="movies", log_y=True, template="simple_white", height=340,
                 title=f"{field.capitalize()}, {tier}: how far apart the two sources are (0 = same figure)",
                 color_discrete_sequence=[HIGH])
    for x in (-np.log10(2), np.log10(2)):
        fig.add_vline(x=x, line_dash="dash", line_color=NEUTRAL)
    st.plotly_chart(fig, width="stretch")
    st.caption("Dashed lines mark a factor of two. The y-axis is logarithmic; ratios beyond 100x sit in the end bars.")

    if not TMDB_CSV.exists():
        st.info("The film-by-film view needs `data/raw/features/tmdb_compare.csv`, which holds TMDB's figures and "
                "stays out of the repo. Get the TMDB file and run `scripts/imdb/tmdb_compare.py` to make it.")
        return
    d = table(TMDB_CSV)
    if tier != "all clean":
        d = d[d.tier == tier] if tier == "100k+" else d[d.tier != "100k+"]
    w, t, ratio = TMDB_FIELDS[field]
    p = d[d[t] > 0]
    st.subheader("Film by film")
    if p.empty:
        st.caption("No films with a TMDB figure in this selection.")
        return
    lim = [float(min(p[w].min(), p[t].min())), float(max(p[w].max(), p[t].max()))]
    sc = px.scatter(p, x=w, y=t, log_x=True, log_y=True, hover_name="title", hover_data=["tconst", "year", "tier"],
                    opacity=.45, template="simple_white", height=520, color_discrete_sequence=[HIGH],
                    labels={w: f"Wikipedia {field} (USD)", t: f"TMDB {field} (USD)"})
    sc.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=LOW, dash="dash"), name="same figure",
                            hoverinfo="skip"))
    st.plotly_chart(sc, width="stretch")
    k = st.slider("Largest disagreements to list", 10, 200, 30, key="tmdb_k")
    worst = p.assign(times_apart=np.exp(np.log(p[ratio]).abs())).nlargest(k, "times_apart")
    show_df(worst[["tconst", "title", "year", "tier", w, t, "times_apart"]], key="tmdb_worst",
            caption="Read these against the infobox before trusting either side. A TMDB figure a thousand or a "
                    "million times too small is usually one typed in thousands or millions.")
