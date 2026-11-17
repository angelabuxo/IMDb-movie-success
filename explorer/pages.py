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
from .common import (arrow_safe, DATA, FEATURES, OUTPUT_DIRS, REPO, TABULAR, flatten_json, log, md, missing, mtime, render_step,
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
- **Screenplay text**: every step of the word-comparison notebook series, rendered from its saved
  outputs. Tables are sortable and filterable, figures are interactive.
- **Network**: the four network analyses (return similarity, pair returns, team history, movers), the per-film
  network features, and a lookup for any film's crew and the films it is linked to.
- **Embedding maps**: every vector space (screenplays, plots, reviews, tags, cast and crew, arcs) in 2D by
  PCA, Isomap, t-SNE and UMAP, coloured by return, genre or year, with a film's nearest films and how well each map
  keeps the space.
- **Embedding map checks**: whether those maps hold up: stability across seeds and settings, whether the spaces
  agree on which films are alike, clusters against profit, sorting by release year, and the hand-check sheet.
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

    num = [c for c in ["log_budget", "log_multiple", "budget_usd", "gross_usd", "multiple", "year", "votes", "rating", "rt", "mc"]
           if c in d and d[c].notna().any()]
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


# ---------------------------------------------------------------- tags

TAG_FILES = ("tag_relevance.npy", "tag_relevance_ids.csv", "tag_relevance_tags.csv", "genre_shares.csv")
TAG_CHECKS = REPO / "analysis" / "tags" / "tag_shares.json"
GENOME, PLOT = "Tag Genome (films to 2014)", "Plot summaries (all years)"
# source -> (matrix, its row ids, the per-film genre table); the tag names are shared
SOURCE_FILES = {GENOME: ("tag_relevance.npy", "tag_relevance_ids.csv", "genre_shares.csv"),
                PLOT: ("plot_tag_relevance.npy", "plot_tag_relevance_ids.csv", "plot_genre_shares.csv")}
PLOT_RECOVERY = REPO / "analysis" / "tags" / "plot_tags_tags.csv"


def source_ready(source: str) -> bool:
    return all((FEATURES / n).exists() for n in SOURCE_FILES[source] + ("tag_relevance_tags.csv",))


@st.cache_data(show_spinner="Reading tag scores…", max_entries=4)
def tag_data(folder: str, stamp: float, source: str = GENOME):
    """Films x tags relevance, the tag names and the per-film genre table (same row order as the matrix)."""
    f = type(FEATURES)(folder)
    matrix, id_file, table_file = SOURCE_FILES[source]
    rel = np.load(f / matrix)
    tags = pd.read_csv(f / "tag_relevance_tags.csv", keep_default_na=False)["tag"].astype(str).tolist()
    films = pd.read_csv(f / table_file)
    ids = pd.read_csv(f / id_file)["tconst"]
    films = films.drop_duplicates("tconst").set_index("tconst").reindex(ids).reset_index()
    if "tier" not in films:
        films["tier"] = ""
    return rel, tags, films


@st.cache_data(show_spinner=False)
def scored_ids(source: str, stamp: float) -> frozenset:
    p = FEATURES / SOURCE_FILES[source][1]
    return frozenset(pd.read_csv(p)["tconst"]) if p.exists() else frozenset()


@st.cache_data(show_spinner=False)
def plot_recovery(stamp: float) -> dict:
    """How well a plot recovers each tag: Pearson r with the Tag Genome score, out of fold (plot_tags.py)."""
    if not PLOT_RECOVERY.exists():
        return {}
    d = pd.read_csv(PLOT_RECOVERY, keep_default_na=False)
    return dict(zip(d["tag"].astype(str), d["recovery_r"].astype(float)))


def tag_shares(rel: np.ndarray, power: float) -> np.ndarray:
    """Relevance to shares that sum to 1 per film: relevance ** power over its row sum."""
    p = np.clip(rel, 0, None).astype(float) ** power
    tot = p.sum(axis=1, keepdims=True)
    return np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)


def tags():
    st.title("Tags and genres")
    st.caption(md("How much each film is each tag. Relevance runs from 0 (doesn't apply) to 1 (applies strongly), for "
                  "1,084 tags, from the MovieLens Tag Genome. A genre share is a film's relevance for one genre, raised "
                  "to a power and divided by its total over the genres, so a film's shares add up to 100%. "
                  "`notebooks/tag_shares.ipynb` has the checks in full."))
    missing = [n for n in TAG_FILES if not (FEATURES / n).exists()]
    if missing:
        st.warning("Needs the tag files in data/raw/features (" + ", ".join(missing) + "). They come in "
                   "`data/bundles/tag-features-2026-11-16.tar.zst`: run `./setup.sh --data`. Or build them with "
                   "`python3 scripts/get_data.py genome`, then from `data/raw`: "
                   "`../../.venv/bin/python ../../scripts/reviews/tag_shares.py`.")
        return
    sources = [GENOME] + ([PLOT] if source_ready(PLOT) else [])
    views = ["One film", "One tag", "Genre mix", "Profit", "Checks"]
    # a click elsewhere (a table row's "Its tags" button, the node explorer) asks for a film and a view; a film the
    # Tag Genome doesn't cover opens on the plot-based scores
    want = st.session_state.pop("tag_film_pending", None)
    if "tag_view_pending" in st.session_state:
        st.session_state["tag_view"] = st.session_state.pop("tag_view_pending")
    if st.session_state.get("tag_view") not in views:
        st.session_state["tag_view"] = views[0]
    if st.session_state.get("tag_source") not in sources:
        st.session_state["tag_source"] = GENOME
    if want and PLOT in sources:
        on = {s: want in scored_ids(s, mtime(FEATURES / SOURCE_FILES[s][1])) for s in sources}
        if not on[st.session_state["tag_source"]] and on[PLOT]:
            st.session_state["tag_source"] = PLOT
    with st.sidebar:
        st.subheader("Tag scores")
        source = st.radio("Scores from", sources, key="tag_source",
                          help="Tag Genome: predicted by MovieLens from user tags, ratings and IMDb reviews, so they carry the "
                               "size of a film's audience; films to 2014. Plot summaries: the same 1,084 tags predicted from "
                               "the film's Wikipedia plot alone (scripts/reviews/plot_tags.py); every year, rougher per film.")
        if PLOT not in sources:
            st.caption("Plot-based scores need `plot_tag_relevance.npy`: run `./setup.sh --data` (plot-tags bundle).")
    fixed = st.session_state["tag_view"] in ("Profit", "Checks")     # these views report the Tag Genome analysis
    use = GENOME if fixed else source
    rel, tag_names, films = tag_data(str(FEATURES), max(mtime(FEATURES / n) for n in SOURCE_FILES[use]), use)
    genres = [c[4:] for c in films.columns if c.startswith("rel_")]
    g_rel = films[["rel_" + g for g in genres]].to_numpy(dtype=float)
    recovery = plot_recovery(mtime(PLOT_RECOVERY)) if use == PLOT else {}
    with st.sidebar:
        if use == PLOT and recovery:
            min_r = st.slider("Only tags a plot recovers, r at least", 0.0, 0.7, 0.4, 0.05, key="tag_min_r",
                              help="Pearson r between the plot-predicted and the Tag Genome score of a tag, out of fold. "
                                   "Below about 0.4 a predicted tag is mostly noise. Genre shares use every genre tag.")
        else:
            min_r = None
        st.subheader("Genre shares")
        power = st.slider("Power", 1.0, 4.0, 2.0, 0.5, key="tag_power",
                          help="Share = relevance ** power ÷ the film's total. 1 spreads a film thinly, because genres "
                               "that don't apply still score a little. Higher powers push the share onto the top genres. "
                               "The order of a film's genres never changes. The saved files use 2.")
    share = tag_shares(g_rel, power)
    imdb = np.array([[g in str(x).split(",") for g in genres] for x in films["genres"]])
    label = (films["primaryTitle"].astype(str) + " (" + films["y"].astype("Int64").astype(str) + ")").tolist()
    med = np.median(rel, axis=0)
    usable = np.array([recovery.get(t, 0.0) >= min_r for t in tag_names]) if min_r is not None else np.ones(len(tag_names), bool)
    if use == GENOME:
        st.caption(f"{len(films):,} films with Tag Genome scores, {len(tag_names):,} tags. Nothing released after 2014 is covered."
                   + (" The Profit and Checks views always use these scores." if fixed and source == PLOT else ""))
    else:
        st.caption(md(f"{len(films):,} films with a plot summary, every year. Each film's tags are predicted from its plot "
                      f"alone, so they are free of audience size and rougher per film. Predicted scores are pulled towards the middle "
                      "(they rarely pass 0.7), so compare films within a tag rather than against a fixed cut-off. {int(usable.sum()):,} of "
                      f"{len(tag_names):,} tags are recovered at r ≥ {min_r:.2f}; the rest are hidden from the tag lists."))

    row_of = {t: k for k, t in enumerate(films["tconst"])}
    if want in row_of:
        st.session_state["tag_film"], st.session_state["tag_film_id"] = row_of[want], want
    view = st.segmented_control("View", views, key="tag_view", label_visibility="collapsed") or views[0]

    def card(k: int, key: str, tags_button: bool):
        """The film's overview, with its three largest shares and five most distinctive tags."""
        g3 = ", ".join(f"{genres[j]} {100 * share[k, j]:.0f}%" for j in np.argsort(-share[k])[:3])
        t5 = ", ".join(tag_names[j] for j in np.argsort(-np.where(usable, rel[k] - med, -9))[:5])
        gp.film_overview(films["tconst"][k], key, note=f"**Genre shares** {g3} · **Tags** {t5}", tags_button=tags_button,
                         fallback={"title": films["primaryTitle"][k], "year": films["y"][k], "genres": films["genres"][k]})

    if view == "One film":
        default = label.index("Alien (1979)") if "Alien (1979)" in label else 0
        if st.session_state.get("tag_film") not in range(len(label)) or st.session_state.get("tag_film_source") != use:
            st.session_state["tag_film"] = row_of.get(st.session_state.get("tag_film_id"), default)
        i = st.selectbox("Film (type to search)", range(len(label)), format_func=lambda k: label[k], key="tag_film")
        st.session_state["tag_film_id"], st.session_state["tag_film_source"] = films["tconst"][i], use   # kept across a source switch
        card(i, "tag_one", tags_button=False)
        l, r = st.columns(2)
        d = pd.DataFrame({"genre": genres, "share": 100 * share[i], "relevance": g_rel[i],
                          "IMDb lists it": np.where(imdb[i], "yes", "no")})
        d = d[d["share"] >= 0.5].sort_values("share")
        l.plotly_chart(px.bar(d, x="share", y="genre", orientation="h", color="IMDb lists it", hover_data=["relevance"],
                              color_discrete_map={"yes": HIGH, "no": NEUTRAL}, template="simple_white", height=420,
                              category_orders={"genre": d["genre"].tolist()[::-1]},
                              title="Genre shares (%), genres under 0.5% left out"), width="stretch", key="tag_film_genres")
        n = r.slider("How many tags", 5, 40, 15, key="tag_film_n")
        rank = r.radio("Rank tags by", ["most distinctive", "highest relevance"], horizontal=True, key="tag_film_rank",
                       help="Most distinctive ranks by relevance minus the tag's median over all films, so tags that score "
                            "high on nearly every film (such as \"original\") drop out.")
        top = np.argsort(-np.where(usable, rel[i] - med if rank == "most distinctive" else rel[i], -9))[:n]
        t = pd.DataFrame({"tag": [tag_names[j] for j in top], "relevance": rel[i, top], "typical film": med[top]})
        fig = go.Figure()
        fig.add_bar(y=t["tag"], x=t["relevance"], orientation="h", name="this film", marker_color=HIGH)
        fig.add_scatter(y=t["tag"], x=t["typical film"], mode="markers", name="typical film (median)",
                        marker=dict(symbol="line-ns-open", size=14, color=LOW, line=dict(width=2)))
        fig.update_layout(template="simple_white", height=max(360, 24 * n + 120), xaxis=dict(range=[0, 1], title="relevance"),
                          yaxis=dict(autorange="reversed"), legend=dict(orientation="h", y=1.08), margin=dict(t=40))
        r.plotly_chart(fig, width="stretch", key="tag_film_tags")
        with st.expander("Every tag for this film"):
            every = pd.DataFrame({"tag": tag_names, "relevance": rel[i].round(3), "typical film": med.round(3),
                                  "lift": (rel[i] - med).round(3)})
            if recovery:
                every["recovered from plots, r"] = [round(recovery.get(t, np.nan), 3) for t in tag_names]
            show_df(every.sort_values("lift", ascending=False), key="tag_film_all", height=360)

    elif view == "One tag":
        choices = [t for t, u in zip(tag_names, usable) if u]
        default = "time travel" if "time travel" in choices else choices[0]
        if any(t not in choices for t in st.session_state.get("tag_pick", [])):
            st.session_state["tag_pick"] = [t for t in st.session_state["tag_pick"] if t in choices] or [default]
        picked = st.multiselect("Tags (several: films are ranked by their lowest score among them)", choices,
                                default=[default], key="tag_pick",
                                format_func=lambda t: f"{t}  (plot r {recovery[t]:.2f})" if t in recovery else t)
        if picked:
            cols = [tag_names.index(t) for t in picked]
            score = rel[:, cols].min(axis=1)
            yr = st.slider("Release year", int(films["y"].min()), int(films["y"].max()),
                           (int(films["y"].min()), int(films["y"].max())), key="tag_year")
            keep = films["y"].between(*yr).to_numpy()
            l, r = st.columns(2)
            l.plotly_chart(px.histogram(x=score[keep], nbins=40, template="simple_white", height=340,
                                        labels={"x": "relevance"}, color_discrete_sequence=[HIGH],
                                        title=f"Relevance over {int(keep.sum()):,} films").update_layout(yaxis_title="films"),
                           width="stretch", key="tag_hist")
            by = pd.DataFrame({"year": films["y"][keep], "strong": score[keep] >= 0.5})
            by = by.groupby(by["year"] // 5 * 5)["strong"].agg(["mean", "size"]).reset_index()
            by = by[by["size"] >= 20]
            r.plotly_chart(px.bar(by, x="year", y="mean", template="simple_white", height=340, hover_data=["size"],
                                  color_discrete_sequence=[NEUTRAL], title="Share of films scoring 0.5 or more, by 5 years")
                           .update_layout(yaxis_tickformat=".0%", yaxis_title=None), width="stretch", key="tag_by_year")
            c = st.columns(3)
            c[0].metric("Median relevance", f"{np.median(score[keep]):.3f}")
            c[1].metric("Films at 0.5 or more", f"{(score[keep] >= 0.5).mean():.1%}")
            c[2].metric("Films at 0.8 or more", f"{(score[keep] >= 0.8).mean():.1%}")
            out = films.loc[keep, ["tconst", "primaryTitle", "y", "genres", "tier"]].assign(relevance=score[keep].round(3))
            slot = st.container()                      # the clicked film shows above the table, where the eye is
            hit = show_df(out.sort_values("relevance", ascending=False).head(200), key="tag_top_films", select=True,
                          caption="The 200 films that score highest. Click a row to see the film.", height=420)
            if hit is not None:
                with slot:
                    card(row_of[hit["tconst"]], "tag_top", tags_button=True)

    elif view == "Genre mix":
        g = st.selectbox("Genre", genres, index=genres.index("Comedy") if "Comedy" in genres else 0, key="tag_genre")
        j = genres.index(g)
        d = pd.DataFrame({"share": 100 * share[:, j], "IMDb lists it": np.where(imdb[:, j], "yes", "no")})
        l, r = st.columns(2)
        l.plotly_chart(px.violin(d, x="share", y="IMDb lists it", color="IMDb lists it", box=True, points=False,
                                 orientation="h", category_orders={"IMDb lists it": ["yes", "no"]},
                                 color_discrete_map={"yes": HIGH, "no": NEUTRAL}, template="simple_white", height=380,
                                 title=f"{g} share (%), films IMDb does and doesn't call {g}")
                       .update_traces(spanmode="hard", scalemode="width").update_layout(showlegend=False),
                       width="stretch", key="tag_mix_hist")
        eff = np.exp(-(share * np.log(np.clip(share, 1e-12, None))).sum(axis=1))
        r.plotly_chart(px.histogram(x=eff, nbins=50, template="simple_white", height=380, color_discrete_sequence=[NEUTRAL],
                                    labels={"x": "effective number of genres"},
                                    title="How spread out films are (1 = one genre holds everything)")
                       .update_layout(yaxis_title="films"), width="stretch", key="tag_mix_eff")
        top = share.argmax(axis=1)
        c = st.columns(3)
        c[0].metric("Top genre is an IMDb genre", f"{imdb[np.arange(len(top)), top].mean():.1%}")
        c[1].metric("Share on IMDb's genres", f"{(share * imdb).sum(axis=1).mean():.1%}")
        c[2].metric("Effective genres (median)", f"{np.median(eff):.2f}")
        out = films[["tconst", "primaryTitle", "y", "genres"]].assign(
            **{f"{g} share %": (100 * share[:, j]).round(1), f"{g} relevance": g_rel[:, j].round(3),
               "top genre": [genres[k] for k in top], "effective genres": eff.round(2)})
        only = st.radio("Show", [f"IMDb calls it {g}", f"IMDb doesn't call it {g}", "all films"], horizontal=True, key="tag_mix_only")
        out = out[imdb[:, j]] if only.startswith("IMDb calls") else out[~imdb[:, j]] if only.startswith("IMDb doesn't") else out
        slot = st.container()
        hit = show_df(out.sort_values(f"{g} share %", ascending=False), key="tag_mix_films", height=420, select=True,
                      caption=f"Sort by the share to find the films IMDb calls {g} that barely are, and the reverse. "
                              "Click a row to see the film.")
        if hit is not None:
            with slot:
                card(row_of[hit["tconst"]], "tag_mix", tags_button=True)

    elif view == "Profit":
        _tag_profit(rel, tag_names, films, genres, g_rel, card, row_of)

    else:
        if not TAG_CHECKS.exists():
            missing("No saved checks (analysis/tags/tag_shares.json). Run scripts/reviews/tag_shares.py.")
            return
        res = json.loads(TAG_CHECKS.read_text())
        il, sv, tm = res["imdb_labels"], res["survey"], res["two_methods"]
        st.caption(md(f"From `analysis/tags/tag_shares.json`, computed at power {res['power']}. Source: {res['source']}."))
        c = st.columns(4)
        c[0].metric("Mean AUC against IMDb genres", f"{il['auc_mean']:.3f}", help="0.5 is a coin flip, 1 is perfect.")
        c[1].metric("Top genre is an IMDb genre", f"{il['fit']['top_genre_in_imdb']:.1%}",
                    help=f"With films shuffled: {il['top_genre_in_imdb_null_mean']:.1%} on average, {il['top_genre_in_imdb_null_max']:.1%} at most.")
        c[2].metric("Spearman with survey answers", f"{sv['mean']['spearman']:.3f}",
                    help="In-sample: the scores were trained on these answers.")
        c[3].metric("Held-out error, 1-5 scale", f"{sv['held_out_mae_5_point_scale']['tagdl']:.2f}",
                    help="The dataset's own ten-fold test of TagDL. A single film-tag score is rough.")
        per = pd.DataFrame(il["per_genre"]).T.reset_index(names="genre")
        per = per.drop(columns=["share_p10_p50_p90_labelled"]).sort_values("auc")
        fig = _null_chart(per, "genre", "auc", "auc_null_max", "AUC of relevance against IMDb's genre label")
        fig.data[1].name = f"highest of {res['null_shuffles']} label shuffles"
        st.plotly_chart(fig.update_yaxes(range=[0.4, 1]), width="stretch", key="tag_auc")
        show_df(per, key="tag_checks_genres", caption="Per genre")
        base = res["tag_baselines"]
        st.subheader("The two scoring methods")
        show_df(pd.DataFrame({
            "scores": ["mean (used)", "TagDL", "glmer"],
            "mean AUC": [il["auc_mean"], il["auc_mean_tagdl"], il["auc_mean_glmer"]],
            "top genre is an IMDb genre": [il["fit"]["top_genre_in_imdb"], il["fit_tagdl"]["top_genre_in_imdb"],
                                           il["fit_glmer"]["top_genre_in_imdb"]],
            "Spearman with survey": [sv[k]["spearman"] for k in ("mean", "tagdl", "glmer")],
            "tags strong on over half of films": [len(base[k]["tags_strong_on_over_half_of_films"]) for k in ("mean", "tagdl", "glmer")],
        }), key="tag_checks_methods")
        st.caption(md(f"TagDL and glmer give the same top genre for {tm['same_top_genre']:.1%} of films, and their scores for a tag "
                      f"correlate at a median of {tm['r_per_tag_median']:.2f} across films. Tags that still score 0.5 or more on "
                      "over half of films under the mean, so read them as noise: "
                      + ", ".join(base["mean"]["tags_strong_on_over_half_of_films"]) + "."))
        cov = pd.DataFrame(res["coverage"]["by_decade"]).T.reset_index(names="decade")
        cov = cov[cov["decade"].astype(int) >= 1950].assign(without=lambda x: x["movies"] - x["scored"])
        st.subheader("Coverage")
        st.plotly_chart(px.bar(cov, x="decade", y=["scored", "without"], template="simple_white", height=320,
                               color_discrete_map={"scored": HIGH, "without": NEUTRAL},
                               title=f"{res['coverage']['scored']:,} of {res['coverage']['movies']:,} films have tag scores")
                        .update_layout(yaxis_title="films", legend_title=None), width="stretch", key="tag_cov")
        st.caption("The scores come from what viewers tagged, rated and wrote after release. Tags such as "
                   "\"oscar (best picture)\", \"boring\" or \"imdb top 250\" record how a film was received, so they can't "
                   "stand in as predictors of profit.")


# ---------------------------------------------------------------- release calendar

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
INFOBOX = [FEATURES / "wiki_infobox.csv", FEATURES / "wiki_infobox_10k.csv"]
# name -> (column, how to combine a cell's films, value where the colour is neutral, green when high, number format)
CAL_METRICS = {
    "Mean adjusted return": ("adjusted", "mean", 0.0, True, "+.2f"),
    "Median gross ÷ budget": ("multiple", "median", 1.0, True, ".2f"),
    "Share grossing under budget": ("under", "mean", 0.5, False, ".0%"),
    "Median budget ($M)": ("budget_m", "median", None, True, ",.0f"),
    "Total gross ($M)": ("gross_m", "sum", None, True, ",.0f"),
    "Mean IMDb rating": ("rating", "mean", None, True, ".2f"),
    "Number of films": ("one", "sum", None, True, ",.0f"),
}


@st.cache_data(show_spinner="Reading release months…", max_entries=2)
def release_months(stamp: float) -> pd.Series:
    """Month of first showing per film, parsed from the Wikipedia infobox (scripts/imdb/parse_infobox.py)."""
    parts = [pd.read_csv(p, usecols=["tconst", "release_month"]) for p in INFOBOX if p.exists()]
    if not parts:
        return pd.Series(dtype=float)
    d = pd.concat(parts).dropna().drop_duplicates("tconst")
    return d.set_index("tconst")["release_month"].astype(int)


def calendar_films() -> pd.DataFrame:
    """Films with a release month, and the adjusted return (ln gross ÷ budget net of ln budget and decade,
    fitted on the clean films from 1975) where it can be computed."""
    m = movies()
    m = m.assign(month=m["tconst"].map(release_months(sum(mtime(p) for p in INFOBOX))))
    m = m[m["month"].between(1, 12) & m["year"].notna()].copy()
    m["month"], m["year"] = m["month"].astype(int), m["year"].astype(int)
    fit = m["clean"] & (m["year"] >= 1975) & (m["multiple"] > 0) & (m["budget_usd"] > 0)
    m["adjusted"] = np.nan
    if fit.sum() > 20:
        f = m[fit]
        decs = sorted(set(f["year"] // 10))[1:]
        x = np.column_stack([np.ones(len(f)), np.log(f["budget_usd"])] + [(f["year"] // 10 == d).astype(float) for d in decs])
        y = np.log(f["multiple"].to_numpy())
        m.loc[fit, "adjusted"] = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    m["under"] = np.where(m["multiple"].notna(), (m["multiple"] < 1).astype(float), np.nan)
    m["budget_m"], m["gross_m"], m["one"] = m["budget_usd"] / 1e6, m["gross_usd"] / 1e6, 1.0
    return m


def calendar():
    st.title("Release calendar")
    st.caption(md("Every film by the year and month it was first shown, one square per month. Green is above the neutral value "
                  "of the chosen measure and red below. The month comes from the Wikipedia infobox and is the **first showing**, "
                  "so festival premieres pile up in September. Gross over budget isn't profit."))
    if not (GRAPH / "nodes.csv").exists() or not any(p.exists() for p in INFOBOX):
        st.warning("Needs the movie graph and the parsed infoboxes (data/raw/features/wiki_infobox*.csv).")
        return
    m = calendar_films()
    if m.empty:
        st.info("No film has a release month.")
        return
    with st.sidebar:
        st.subheader("Films")
        clean = st.toggle("Clean films only", True, key="cal_clean",
                          help="Usable USD budget and worldwide gross. The money measures need these.")
        tiers = st.multiselect("Vote tier", sorted(m["tier"].dropna().unique()), default=sorted(m["tier"].dropna().unique()), key="cal_tier")
        all_genres = sorted({g for gs in m["genres"].dropna() for g in gs.split(",") if g})
        need = st.multiselect("Must include genre", all_genres, key="cal_genre")
    d = m[m["tier"].isin(tiers)]
    if clean:
        d = d[d["clean"]]
    for g in need:
        d = d[d["genres"].fillna("").str.split(",").apply(lambda v, g=g: g in v)]
    if d.empty:
        st.info("No films match the filters.")
        return
    c = st.columns([2, 3])
    metric = c[0].selectbox("Measure", list(CAL_METRICS), key="cal_metric")
    lo, hi = int(d["year"].min()), int(d["year"].max())
    years = c[1].slider("Years", lo, hi, (max(lo, min(1990, hi)), hi), key="cal_years") if lo < hi else (lo, hi)
    c = st.columns([4, 1])
    months = c[0].pills("Months", range(1, 13), selection_mode="multi", default=list(range(1, 13)),
                        format_func=lambda k: MONTHS[k - 1], key="cal_months") or list(range(1, 13))
    min_n = c[1].number_input("Min films per square", 1, 50, 3, key="cal_min",
                              help="Squares with fewer films are left grey: one film isn't a pattern.")
    col, how, neutral, green_high, fmt = CAL_METRICS[metric]
    d = d[d["year"].between(*years) & d["month"].isin(months)]
    d = d[d[col].notna()]
    st.caption(f"{len(d):,} films in the selection" + (" with this measure" if col not in ("one",) else ""))
    if d.empty:
        return
    cell = d.groupby(["year", "month"]).agg(value=(col, how), films=("one", "sum"),
                                            top=("title", lambda t: ", ".join(t.head(3)))).reset_index()
    shown = cell[cell["films"] >= min_n]
    faint = cell[cell["films"] < min_n]
    centre = neutral if neutral is not None else (float(shown["value"].median()) if len(shown) else 0.0)
    spread = float(np.nanpercentile(np.abs(shown["value"] - centre), 95)) if len(shown) else 1.0
    scale = "RdYlGn" if green_high else "RdYlGn_r"
    order = [k for k in range(1, 13) if k in months]
    wide = st.toggle("Years across, months down", True, key="cal_wide", help="Off: one row per year, which reads better on a narrow screen.")
    ax = (lambda df: (df["year"], df["month"].map(lambda k: MONTHS[k - 1]))) if wide else (lambda df: (df["month"].map(lambda k: MONTHS[k - 1]), df["year"]))
    n_years = years[1] - years[0] + 1
    size = int(np.clip((1100 if wide else 520) / (n_years if wide else 12) - 3, 9, 34)) if wide else 26
    fig = go.Figure()
    if len(faint):
        x, y = ax(faint)
        fig.add_scatter(x=x, y=y, mode="markers", name=f"fewer than {min_n} films", hoverinfo="skip",
                        marker=dict(symbol="square", size=size, color="#E6E6E6"))
    x, y = ax(shown)
    fig.add_scatter(x=x, y=y, mode="markers", name=metric, customdata=np.column_stack([shown["year"], shown["month"], shown["films"], shown["top"]]),
                    text=[format(v, fmt) for v in shown["value"]],
                    hovertemplate="%{customdata[0]} " + "%{y}" * wide + "%{x}" * (not wide) + "<br>" + metric + ": %{text}<br>%{customdata[2]} films<br>%{customdata[3]}<extra></extra>",
                    marker=dict(symbol="square", size=size, color=shown["value"], colorscale=scale, cmin=centre - spread, cmax=centre + spread,
                                colorbar=dict(title=None, thickness=12, len=0.8, tickformat=fmt.lstrip("+")), line=dict(width=0)))
    month_axis = dict(categoryorder="array", categoryarray=[MONTHS[k - 1] for k in (order[::-1] if wide else order)], title=None)
    year_axis = dict(dtick=5 if n_years > 25 else 1, title=None, **({} if wide else {"autorange": "reversed"}))
    fig.update_layout(template="simple_white", showlegend=False, margin=dict(t=20, b=30, l=40, r=10),
                      height=(size + 5) * len(order) + 90 if wide else max(320, 30 * n_years + 80),
                      xaxis=year_axis if wide else month_axis, yaxis=month_axis if wide else year_axis)
    ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points",), key="cal_grid")
    st.caption(md(f"Neutral colour at {format(centre, fmt)}" + ("" if neutral is not None else " (the median square)")
                  + f". Grey squares have fewer than {min_n} films. Click a square to list its films."))

    l, r = st.columns(2)
    agg = lambda by: d.groupby(by).agg(value=(col, how), films=("one", "sum")).reset_index()
    bm = agg("month").assign(month=lambda t: t["month"].map(lambda k: MONTHS[k - 1]))
    colour = lambda t: np.where((t["value"] >= centre) == green_high, "#2E8B57", "#C0392B") if neutral is not None else HIGH
    fig_m = go.Figure(go.Bar(x=bm["month"], y=bm["value"] - (centre if neutral is not None else 0), base=centre if neutral is not None else 0,
                             marker_color=colour(bm), customdata=bm["films"], text=[format(v, fmt) for v in bm["value"]], textposition="none",
                             hovertemplate="%{x}: %{text}<br>%{customdata} films<extra></extra>"))
    l.plotly_chart(fig_m.update_layout(template="simple_white", height=320, title=f"{metric} by month, {years[0]} to {years[1]}",
                                       yaxis_tickformat=fmt.lstrip("+"), margin=dict(t=50)), width="stretch", key="cal_by_month")
    by = agg("year")
    fig_y = go.Figure(go.Bar(x=by["year"], y=by["value"] - (centre if neutral is not None else 0), base=centre if neutral is not None else 0,
                             marker_color=colour(by), customdata=by["films"], text=[format(v, fmt) for v in by["value"]], textposition="none",
                             hovertemplate="%{x}: %{text}<br>%{customdata} films<extra></extra>"))
    r.plotly_chart(fig_y.update_layout(template="simple_white", height=320, title=f"{metric} by year, the months selected",
                                       yaxis_tickformat=fmt.lstrip("+"), margin=dict(t=50)), width="stretch", key="cal_by_year")

    pts = ev.selection["points"] if ev and getattr(ev, "selection", None) else []
    picked = [(int(p["customdata"][0]), int(p["customdata"][1])) for p in pts if p.get("customdata")]
    st.subheader("Films in a square")
    c = st.columns(2)
    ys = sorted(d["year"].unique(), reverse=True)
    if picked:
        st.session_state["cal_pick_year"], st.session_state["cal_pick_month"] = picked[0]
    if st.session_state.get("cal_pick_year") not in ys:
        st.session_state["cal_pick_year"] = ys[0]
    if st.session_state.get("cal_pick_month") not in order:
        st.session_state["cal_pick_month"] = order[0]
    py = c[0].selectbox("Year", ys, key="cal_pick_year")
    pm = c[1].selectbox("Month", order, format_func=lambda k: MONTHS[k - 1], key="cal_pick_month")
    f = d[(d["year"] == py) & (d["month"] == pm)]
    slot = st.container()
    cols = ["tconst", "title", "genres", "budget_usd", "gross_usd", "multiple", "adjusted", "rating", "votes"]
    hit = show_df(f.sort_values("gross_usd", ascending=False)[cols].round({"adjusted": 3, "multiple": 2}), key="cal_films", select=True,
                  caption=f"{len(f)} films first shown in {MONTHS[pm - 1]} {py}. Click a row to see the film.", height=320)
    if hit is not None:
        with slot:
            gp.film_overview(hit["tconst"], "cal_film", tags_button=hit["tconst"] in gp._tagged_films(mtime(FEATURES / "tag_relevance_ids.csv") + mtime(FEATURES / "plot_tag_relevance_ids.csv")))


FOLLOW_VIEWS = ["Tags from plots", "Usual genre mix", "Seen and liked", "Critics and audience", "Franchise and month"]


def _bar_h(df, x, y, title, colour=None, height=None, lines=()):
    fig = px.bar(df, x=x, y=y, orientation="h", color=colour, template="simple_white",
                 height=height or max(320, 22 * len(df) + 110), title=title,
                 color_discrete_sequence=[HIGH, LOW, NEUTRAL], category_orders={y: df[y].tolist()[::-1]})
    for v in lines:
        fig.add_vline(x=v, line_dash="dash", line_color="#444")
    return fig.update_layout(yaxis_title=None, legend_title=None)


def tag_followups():
    """Five follow-ups to the tag and profit analysis, drawn from the saved results only."""
    st.title("Tags: follow-ups")
    st.caption(md("What came after finding that the Tag Genome scores carry audience size: tags predicted from plot summaries, "
                  "the usual genre mix of directors and studios, seen and liked against earned, critics against audience, and "
                  "franchise and release month as controls. Everything here is read from `analysis/tags/`; "
                  "`notebooks/tag_followups.ipynb` has the full account. Gross over budget isn't profit."))
    pj, fj = TAG_PROFIT / "plot_tags.json", TAG_PROFIT / "tag_followups.json"
    if not pj.exists() or not fj.exists():
        missing("Needs analysis/tags/plot_tags.json and tag_followups.json (scripts/reviews/plot_tags.py and tag_followups.py).")
        return
    plot, fol = json.loads(pj.read_text()), json.loads(fj.read_text())
    if st.session_state.get("fu_view") not in FOLLOW_VIEWS:
        st.session_state["fu_view"] = FOLLOW_VIEWS[0]
    view = st.segmented_control("View", FOLLOW_VIEWS, key="fu_view", label_visibility="collapsed") or FOLLOW_VIEWS[0]

    if view == "Tags from plots":
        rec, leak, ret = plot["recovery"], plot["leak"], plot["return"]
        st.caption(md(f"A ridge regression predicts the 1,084 tags from the embedding of a film's Wikipedia plot summary, trained on the "
                      f"{plot['films_with_plot_and_tag_scores']:,} films that have both. {plot['films_with_plot']:,} films have a plot, "
                      "including the years the Tag Genome doesn't cover."))
        c = st.columns(4)
        c[0].metric("Content tags: median r with the Tag Genome", f"{rec['by_class']['content']['median_r']:.2f}",
                    help="Out of fold: each film's tags come from a model that never saw it.")
        c[1].metric("Genre AUC against IMDb, films after 2014", f"{rec['genre_auc_mean']['plot_films_after_2014']:.3f}",
                    help=f"Tag Genome on its own films: {rec['genre_auc_mean']['tag_genome_scored_films']:.3f}.")
        c[2].metric("Votes predicted from the plot, R²", f"{leak['r2_votes_from_plot_embedding']:.2f}",
                    help=f"From the Tag Genome's content tags: {leak['r2_votes_from_tag_genome_content_tags']:.2f}. This is the audience leak.")
        c[3].metric("Return predicted from the plot, later films, R²", f"{ret['sets']['plot embedding']['trained_before_2005']:.3f}",
                    help="Trained on films before 2005, tested on films from 2005.")
        rows = [{"features": k, "count": v["features"], "random folds": v["random_folds"], "trained before 2005": v["trained_before_2005"],
                 "trained to 2014, tested after": v["trained_to_2014_tested_after"], "votes fixed, trained before 2005": v["votes_fixed_trained_before_2005"],
                 "highest with outcome shuffled": v["null_max"]} for k, v in ret["sets"].items()]
        st.subheader("How much of the adjusted return content accounts for")
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
        sc = ret["scored_films"]
        st.caption(md(f"Out-of-sample R² over {ret['films']:,} clean films. On the {sc['films']:,} films with Tag Genome scores, trained before "
                      f"2005: Tag Genome content tags {sc['Tag Genome content tags']['trained_before_2005']:.3f}, plot embedding "
                      f"{sc['plot embedding']['trained_before_2005']:.3f}, IMDb genre labels {sc['IMDb genre labels']['trained_before_2005']:.3f}."))
        if (TAG_PROFIT / "plot_tags_tags.csv").exists():
            t = table(TAG_PROFIT / "plot_tags_tags.csv")
            t = t[(t["class"] == "content") & (t["recovery_r"] > 0.5)].sort_values("rho_adjusted_plot")
            both = pd.concat([t.head(12), t.tail(12)])
            st.plotly_chart(_bar_h(both, "rho_adjusted_plot", "tag", "Plot-predicted tags against the adjusted return (content tags a plot recovers, r over 0.5)",
                                   lines=(ret["per_tag"]["familywise_95"], -ret["per_tag"]["familywise_95"])), width="stretch", key="fu_plot_tags")
            ag = ret["per_tag"]["agreement_with_tag_genome"]
            st.caption(md(f"Dashed: the family-wise threshold. On the same films, the plot's rho and the Tag Genome's rho for a tag correlate at "
                          f"{ag['r_between_rhos_recovered_content']:.2f} and share a sign for {ag['same_sign_recovered_content']:.0%} of tags; the "
                          f"median |rho| is {ag['median_abs_rho_plot']:.3f} from plots against {ag['median_abs_rho_tag_genome']:.3f} from the Tag Genome."))
            with st.expander("Every tag: how well a plot recovers it, and its rho with the return"):
                show_df(table(TAG_PROFIT / "plot_tags_tags.csv").sort_values("recovery_r", ascending=False), key="fu_plot_all", height=380)
        s = plot["screenplays"]
        st.subheader("Screenplays, the second source")
        rows = [{"features": k, **{a: b for a, b in v.items()}} for k, v in s["return"].items() if isinstance(v, dict)]
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
        st.caption(md(f"{s['return']['films']:,} clean films with a screenplay. A screenplay recovers a content tag at a median r of "
                      f"{s['recovery_by_class']['content']['median_r']:.2f}, a plot on the same films at "
                      f"{s['plot_on_the_same_films']['by_class']['content']['median_r']:.2f}."))

    elif view == "Usual genre mix":
        st.caption(md("How far a film's genre mix is from the mean mix of its director's or studio's earlier films (1 minus the cosine), "
                      "against the adjusted return. Only earlier films are used."))
        rows, quart = [], []
        for vec, by_role in fol["usual_mix"].items():
            for role, o in by_role.items():
                tr = o.get("track_record", {})
                rows.append({"mix measured by": vec, "whose": role, "films": o["films"], "rho with return": o["against_adjusted"]["rho"],
                             "rho, film's genres held fixed": o["against_adjusted_genre_fixed"]["rho"], "p": o["against_adjusted_genre_fixed"]["p"],
                             "track record when staying": tr.get("rho_track_when_staying"), "when leaving": tr.get("rho_track_when_leaving"),
                             "p of the difference": tr.get("p")})
                quart += [{"case": f"{role}, {vec.split(' (')[0]}", "quartile of departure": str(q["quartile"]),
                           "mean adjusted return": q["mean_adjusted_genre_fixed"], "films": q["films"],
                           "median gross ÷ budget": q["median_multiple"]} for q in o["by_quartile"]]
        st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
        st.plotly_chart(px.bar(pd.DataFrame(quart), x="quartile of departure", y="mean adjusted return", color="case", barmode="group",
                               template="simple_white", height=380, hover_data=["films", "median gross ÷ budget"],
                               title="Adjusted return (film's own genres held fixed) by how far the film is from the usual mix: 1 = closest")
                        .add_hline(y=0, line_color="#444"), width="stretch", key="fu_mix")
        st.caption("The IMDb-label rows are the safe ones: the Tag Genome shares were measured after release. "
                   "A studio that leaves its usual mix may also be a studio in trouble, so this is an association.")

    elif view == "Seen and liked":
        a = fol["audience"]
        c = st.columns(4)
        c[0].metric("Votes against return, rho", f"{a['votes_vs_adjusted']['rho']:+.2f}")
        c[1].metric("Rating against return, rho", f"{a['rating_vs_adjusted']['rho']:+.2f}")
        c[2].metric("Rating against revenue, rho", f"{a['rating_vs_gross']['rho']:+.2f}")
        c[3].metric("Rating against budget, rho", f"{a['rating_vs_budget']['rho']:+.2f}")
        q = pd.DataFrame({"films": list(a["quadrants"]), "share": list(a["quadrants"].values())})
        l, r = st.columns([1, 2])
        l.dataframe(q.assign(share=(100 * q["share"]).round(1)), width="stretch", hide_index=True)
        l.caption("Liked: rated above the decade's median. Earned: returned more than budget and decade predict.")
        pick = r.selectbox("Quadrant", list(a["quadrants"]), index=1, key="fu_quad")
        g = pd.DataFrame([{"genre": k, "lift": v["lift"][pick], "films": v["films"], "q": v["lift_q"][pick]} for k, v in a["per_genre"].items()]
                         ).sort_values("lift")
        r.plotly_chart(_bar_h(g, "lift", "genre", f"How over-represented each genre is among films that are: {pick}", lines=(1,))
                       .update_traces(customdata=g[["films", "q"]], hovertemplate="%{y}: lift %{x:.2f}<br>%{customdata[0]} films, q %{customdata[1]:.3f}"),
                       width="stretch", key="fu_lift")
        att = pd.DataFrame([{"genre": k, "votes for its gross": v["mean_attention"], "rating (z within decade)": v["mean_rating_z"],
                             "adjusted return": v["mean_adjusted"], "films": v["films"]} for k, v in a["per_genre"].items()])
        st.plotly_chart(px.scatter(att, x="adjusted return", y="rating (z within decade)", text="genre", size="films", template="simple_white",
                                   height=460, color="votes for its gross", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                                   title="Genres by what they earn and how they are rated; colour is votes relative to gross")
                        .update_traces(textposition="top center").add_hline(y=0, line_color="#bbb").add_vline(x=0, line_color="#bbb"),
                        width="stretch", key="fu_att")

    elif view == "Critics and audience":
        cr = fol["critics"]
        src = st.radio("Critic score", ["metacritic", "rotten_tomatoes"], horizontal=True, key="fu_crit",
                       format_func=lambda k: "Metacritic" if k == "metacritic" else "Rotten Tomatoes")
        c = cr[src]
        m = st.columns(4)
        m[0].metric("Films", f"{c['films']:,}")
        m[1].metric("Critics against return, rho", f"{c['score_vs_adjusted']['rho']:+.2f}")
        m[2].metric("Audience against return, same films", f"{c['rating_vs_adjusted_same_films']['rho']:+.2f}")
        m[3].metric("The gap against return, rho", f"{c['gap_vs_adjusted']['rho']:+.3f}", help=f"p = {c['gap_vs_adjusted']['p']:.3f}")
        g = pd.DataFrame([{"genre": k, "critics minus audience": v["mean_gap"], "films": v["films"], "q": v["q"]} for k, v in c["per_genre"].items()]
                         ).sort_values("critics minus audience")
        st.plotly_chart(_bar_h(g, "critics minus audience", "genre", "Mean gap by genre, in standard deviations within decade (positive: critics kinder than viewers)")
                        .update_traces(customdata=g[["films", "q"]], hovertemplate="%{y}: %{x:+.2f}<br>%{customdata[0]} films, q %{customdata[1]:.3f}"),
                        width="stretch", key="fu_gap")
        tm = cr["two_methods"]
        st.caption(md(f"The two critic scores give gaps that correlate at {tm['rho_between_gaps']:.2f} across {tm['films']:,} films, and the genre "
                      f"means rank alike (rho {tm['rho_per_genre_mean_gaps']:.2f}). The gap falls with budget (rho {c['gap_vs_budget']['rho']:+.2f}) "
                      f"and with votes (rho {c['gap_vs_votes']['rho']:+.2f})."))

    else:
        ct = fol["controls"]
        fr, rm, wc = ct["franchise"], ct["release_month"], ct["with_controls"]
        c = st.columns(4)
        c[0].metric("Films flagged as franchise", f"{fr['flagged']:.1%}", help=f"Follows another film: {fr['follows_another_film']:.1%}.")
        c[1].metric("Flag against return, rho", f"{fr['mean_adjusted']['flag_vs_adjusted']['rho']:+.2f}",
                    help=f"'Follows another film' alone: {fr['mean_adjusted']['follows_vs_adjusted']['rho']:+.2f}. The flag also marks films that later "
                         "got a sequel, which is a result of success.")
        c[2].metric("Return variance: follows a film", f"{wc['by_follows_alone']:.1%}", help=f"The full flag: {wc['by_full_flag_alone']:.1%}.")
        c[3].metric("Return variance: release month", f"{rm['share_of_variance_in_adjusted']:.1%}", help=f"95% of shuffles stay under {rm['share_null_95']:.1%}.")
        tc = pd.DataFrame(fr["tag_check"]).T.reset_index(names="Tag Genome tag")
        st.subheader("The franchise flag against the Tag Genome's own tags")
        st.dataframe(tc.round(3), width="stretch", hide_index=True)
        mo = pd.DataFrame(rm["by_month"]).T.reset_index(names="month")
        st.plotly_chart(px.bar(mo, x="month", y="mean_adjusted", template="simple_white", height=340, hover_data=["films", "median_budget_musd"],
                               color_discrete_sequence=[HIGH], title="Mean adjusted return by release month (first showing, so festivals land in September)")
                        .add_hline(y=0, line_color="#444"), width="stretch", key="fu_month")
        gsh = pd.DataFrame(wc["genre_shares"]).T.reset_index(names="genre").sort_values("after")
        fig = go.Figure()
        fig.add_bar(y=gsh["genre"], x=gsh["before"], orientation="h", name="budget and decade fixed", marker_color=NEUTRAL)
        fig.add_bar(y=gsh["genre"], x=gsh["after"], orientation="h", name="plus follows-a-film and month", marker_color=HIGH)
        st.plotly_chart(fig.update_layout(template="simple_white", height=560, barmode="group", legend=dict(orientation="h", y=1.06),
                                          title="Genre share against the return, before and after the controls", xaxis_title="rho"),
                        width="stretch", key="fu_ctl")
        tg = wc["tags"]
        st.caption(md(f"With 'follows another film' and month held fixed, the 40 strongest content tags all keep their sign "
                      f"({tg['top_40_content_same_sign']:.0%}) and shrink to {tg['top_40_content_mean_shrink']:.0%} of their size on average. "
                      f"The 'sequel' tag goes from {tg['commercial']['sequel']['before']:+.2f} to {tg['commercial']['sequel']['after']:+.2f}."))


TAG_PROFIT = REPO / "analysis" / "tags"
OUTCOMES = {"adjusted": "Return, budget and decade held fixed", "adjusted_votes": "Return, budget, decade and IMDb votes held fixed",
            "multiple": "ln(gross ÷ budget)", "gross": "ln(gross), the revenue", "budget": "ln(budget)"}


def _tag_profit(rel, tag_names, films, genres, g_rel, card, row_of):
    """Tags and genre mixes against revenue and return: the saved results of scripts/reviews/tag_profit.py,
    and the same sample drawn live for one tag or one pair of genres."""
    if not (TAG_PROFIT / "tag_profit.json").exists() or not (GRAPH / "nodes.csv").exists():
        missing("Needs analysis/tags/tag_profit.json (run scripts/reviews/tag_profit.py) and the movie graph.")
        return
    res = json.loads((TAG_PROFIT / "tag_profit.json").read_text())
    per_tag = table(TAG_PROFIT / "tag_profit_tags.csv")
    m = movies().set_index("tconst")
    ok = films["tconst"].map(lambda t: t in m.index and bool(m.at[t, "clean"]) and m.at[t, "year"] >= 1975
                             and pd.notna(m.at[t, "multiple"]) and m.at[t, "multiple"] > 0).to_numpy()
    idx = np.flatnonzero(ok)
    mm = m.loc[films["tconst"][idx]]
    mult, budget, year = mm["multiple"].to_numpy(), mm["budget_usd"].to_numpy(), mm["year"].to_numpy()
    decs = sorted(set(year // 10))[1:]
    x = np.column_stack([np.ones(len(idx)), np.log(budget)] + [(year // 10 == d).astype(float) for d in decs])
    adj = np.log(mult) - x @ np.linalg.lstsq(x, np.log(mult), rcond=None)[0]
    sm = res["sample"]
    st.caption(md(f"{sm['films']:,} clean films from {sm['years'][0]} to {sm['years'][1]} with tag scores. "
                  f"{sm['grossed_under_budget']:.1%} grossed under budget; the median gross ÷ budget is {sm['median_multiple']:.2f}. "
                  "Gross over budget isn't profit: budgets leave out marketing, and cinemas keep part of the gross."))
    st.warning(md("Tag scores were predicted from what viewers tagged, rated and wrote **after** release, so they can carry the "
                  "size of a film's audience, which is the outcome. Read every number here as a description of films that "
                  "did well, not as a way to predict one. Holding IMDb votes fixed is the strict version of each result."))

    st.subheader("Which tags go with money")
    c = st.columns([2, 1, 1])
    out = c[0].selectbox("Outcome", list(OUTCOMES), format_func=OUTCOMES.get, key="tp_outcome")
    kind = c[1].selectbox("Tags", ["content", "reception", "commercial", "all"], key="tp_class",
                          help="Reception tags state an opinion, an award or a list; commercial tags a fact such as "
                               "sequel or big budget. Both record success more than they explain it.")
    k = c[2].slider("How many each way", 5, 30, 12, key="tp_k")
    t = res["tags"][out]
    d = per_tag if kind == "all" else per_tag[per_tag["class"] == kind]
    d = d.sort_values("rho_" + out)
    both = pd.concat([d.head(k), d.tail(k)]).drop_duplicates("tag")
    fig = px.bar(both, x="rho_" + out, y="tag", orientation="h", color="class", template="simple_white",
                 height=max(420, 22 * len(both) + 120), hover_data=["films_high", "q_" + out],
                 color_discrete_map={"content": HIGH, "reception": LOW, "commercial": NEUTRAL},
                 category_orders={"tag": both["tag"].tolist()[::-1]},
                 title=f"Spearman rho between a tag's relevance and: {OUTCOMES[out].lower()}")
    for v in (t["familywise_95"], -t["familywise_95"]):
        fig.add_vline(x=v, line_dash="dash", line_color="#444")
    st.plotly_chart(fig.update_layout(xaxis_title="rho", yaxis_title=None), width="stretch", key="tp_bar")
    st.caption(md(f"Dashed lines: the largest |rho| any tag reaches in 95% of {res['settings']['shuffles']:,} shuffles of the outcome "
                  f"({t['familywise_95']:.3f}). {t['tags_beyond_familywise']:,} of {len(per_tag):,} tags are beyond it, so nearly "
                  f"everything is 'significant'; the size is what matters. The median |rho| is {t['median_abs_rho']:.3f}."))
    with st.expander("Every tag"):
        show_df(d[["tag", "class", "films_high", "median_relevance"] + [c for c in d.columns if c.startswith(("rho_", "q_"))]]
                .sort_values("rho_" + out, ascending=False), key="tp_all", height=380)

    st.subheader("One tag")
    default = tag_names.index("black comedy") if "black comedy" in tag_names else 0
    j = st.selectbox("Tag", range(len(tag_names)), index=default, format_func=lambda i: tag_names[i], key="tp_tag")
    v = rel[idx, j]
    bins = pd.cut(v, [0, 0.2, 0.5, 0.8, 1.0001], right=False, labels=["0 to 0.2", "0.2 to 0.5", "0.5 to 0.8", "0.8 to 1"])
    dd = pd.DataFrame({"relevance": bins, "ln(gross ÷ budget)": np.log(mult), "adjusted return": adj})
    l, r = st.columns(2)
    l.plotly_chart(px.box(dd, x="relevance", y="ln(gross ÷ budget)", template="simple_white", height=360, points=False,
                          category_orders={"relevance": list(bins.categories)}, color_discrete_sequence=[HIGH],
                          title="ln(gross ÷ budget): 0 means it grossed its budget").add_hline(y=0, line_dash="dash"),
                   width="stretch", key="tp_box")
    by = dd.groupby("relevance", observed=False).agg(films=("adjusted return", "size"),
                                                     median_multiple=("ln(gross ÷ budget)", lambda a: float(np.exp(a.median())) if len(a) else np.nan),
                                                     under_budget=("ln(gross ÷ budget)", lambda a: float((a < 0).mean()) if len(a) else np.nan),
                                                     mean_adjusted=("adjusted return", "mean")).reset_index()
    r.dataframe(by.round(3), width="stretch", hide_index=True)
    row = per_tag[per_tag["tag"].astype(str) == tag_names[j]]
    if len(row):
        r.caption(md(f"Class: {row['class'].iloc[0]}. rho with revenue {row['rho_gross'].iloc[0]:+.3f}, with budget "
                     f"{row['rho_budget'].iloc[0]:+.3f}, with the adjusted return {row['rho_adjusted'].iloc[0]:+.3f}, "
                     f"and with IMDb votes also held fixed {row['rho_adjusted_votes'].iloc[0]:+.3f}."))

    st.subheader("Two genres together")
    c = st.columns(3)
    a = c[0].selectbox("Genre A", genres, index=genres.index("Drama") if "Drama" in genres else 0, key="tp_a")
    b = c[1].selectbox("Genre B", [g for g in genres if g != a], key="tp_b")
    cut = c[2].slider("Counts as the genre from relevance", 0.2, 0.9, float(res["settings"]["high"]), 0.1, key="tp_cut")
    ha, hb = g_rel[idx, genres.index(a)] >= cut, g_rel[idx, genres.index(b)] >= cut
    cells = [("neither", ~ha & ~hb), (f"{a} only", ha & ~hb), (f"{b} only", ~ha & hb), (f"{a} and {b}", ha & hb)]
    grid = pd.DataFrame([{"films": name, "count": int(msk.sum()),
                          "median gross ÷ budget": float(np.median(mult[msk])) if msk.any() else np.nan,
                          "grossed under budget": float((mult[msk] < 1).mean()) if msk.any() else np.nan,
                          "median budget ($M)": float(np.median(budget[msk]) / 1e6) if msk.any() else np.nan,
                          "mean adjusted return": float(adj[msk].mean()) if msk.any() else np.nan} for name, msk in cells])
    st.dataframe(grid.round(3), width="stretch", hide_index=True)
    if all(msk.any() for _, msk in cells):
        inter = adj[cells[3][1]].mean() - adj[cells[1][1]].mean() - adj[cells[2][1]].mean() + adj[cells[0][1]].mean()
        st.caption(md(f"Interaction: {inter:+.3f}. That is the adjusted return of films that are both, minus what the two genres add "
                      "separately. Zero means the combination is worth exactly the sum of its parts."))
    pr = res["pairs"]
    st.caption(md(f"Across the {pr['pairs_tested']} pairs with at least {res['settings']['min_cell']} films in every cell, "
                  f"{pr['pairs_q_below_0.05']} have an interaction with q below 0.05 (smallest q {pr['smallest_q']:.2f}), and "
                  f"{pr['pairs_beyond_familywise']} exceed the family-wise threshold of {pr['familywise_95']:.2f}."))
    pairs = table(TAG_PROFIT / "tag_profit_pairs.csv")
    with st.expander("Every pair tested"):
        show_df(pairs.reindex(pairs["interaction"].abs().sort_values(ascending=False).index), key="tp_pairs", height=380)
    both_films = films.iloc[idx[cells[3][1]]][["tconst", "primaryTitle", "y", "genres"]].assign(
        **{"gross ÷ budget": mult[cells[3][1]].round(2), "adjusted return": adj[cells[3][1]].round(3)})
    slot = st.container()
    hit = show_df(both_films.sort_values("gross ÷ budget", ascending=False), key="tp_both", height=320, select=True,
                  caption=f"Films that are both {a} and {b}. Click a row to see the film.")
    if hit is not None:
        with slot:
            card(row_of[hit["tconst"]], "tp_both", tags_button=True)

    st.subheader("How much of the return the tags account for")
    pred = res["prediction"]
    rows = [{"features": name, "count": v["features"], "R² random folds": v["adjusted"]["cv_r2"],
             "R² trained before %d, tested after" % res["settings"]["split_year"]: v["adjusted"]["time_split_r2"],
             "R² random folds, votes held fixed": v["adjusted_votes"]["cv_r2"],
             "R² in time, votes held fixed": v["adjusted_votes"]["time_split_r2"],
             "highest R² with the outcome shuffled": max(v["adjusted"]["cv_r2_null_max"], v["adjusted_votes"]["cv_r2_null_max"])}
            for name, v in pred["sets"].items()]
    st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)
    au = res.get("audience", {})
    if au:
        st.caption(md(f"Out-of-sample R² of a ridge regression of the adjusted return. Budget and decade alone account for "
                      f"{pred['variance_explained_by_budget_and_decade']:.1%} of ln(gross ÷ budget). The audience-size check: the "
                      f"content tags predict IMDb votes (net of budget and decade) with R² {au['cv_r2_content_tags_predicting_votes']:.2f}, "
                      f"and votes go with the adjusted return at rho {au['rho_votes_with_adjusted']:+.2f}. So part of what the tags "
                      "'explain' is how many people saw the film."))


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
        log.exception("can't read %s", f)
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
VERIFIED_STALE: dict[str, str] = {}   # step -> why an older config.yaml can't affect it
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
    for pkg in ("scripts_text",):
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
        sr = rb["scripts_and_reviews"]
        runs = {"reproduce": "committed (Wikipedia)", "wikipedia": "TMDB films, Wikipedia money", "tmdb": "TMDB films, TMDB money"}
        rows = [{"result": f"script: {f}", **{lab: f"{sr[v]['features'][f]['rho']:+.3f} (q {sr[v]['features'][f]['q_partial']:.3f})"
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
        missing("The film-by-film view needs `data/raw/features/tmdb_compare.csv`, which holds TMDB's figures and "
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
