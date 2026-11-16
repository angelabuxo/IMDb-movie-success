"""Embedding maps: every vector space the project builds, flattened to 2D four ways (PCA, Isomap, t-SNE, UMAP).

Reads the coordinates and scores notebooks/embedding_maps.ipynb saves; nothing is recomputed except a film's
nearest films on the map shown. Only titles and numbers are shown, never script, review or plot text.

checks() is the page for notebooks/embedding_map_checks.ipynb. It reads only the committed numbers in
analysis/embedding_maps/, so it works without the data folder. Both pages show the notebooks' own written reading,
read from the .ipynb files, so the text lives in one place.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from . import nav
from .common import FEATURES, REPO, md, missing, show_df, table

MAPS = FEATURES / "embedding_maps.csv"
SCORES = FEATURES / "embedding_map_scores.csv"
COMMITTED = REPO / "analysis" / "embedding_maps"   # numbers only, in git
SCORES_COMMITTED = COMMITTED / "scores.csv"
NB_MAPS = REPO / "notebooks" / "embedding_maps.ipynb"
NB_CHECKS = REPO / "notebooks" / "embedding_map_checks.ipynb"
METHODS = ["PCA", "Isomap", "t-SNE", "UMAP"]
ALL = "All four"
K = 10
COLOURS = {"ln(multiple)": "log_mult", "net multiple": "log_mult_net", "primary genre": "primary_genre",
           "release year": "year"}
# same as analysis/projections.py, so the page and the notebook figures read alike
GENRE_COLOURS = {"Horror": "#d62728", "Sci-Fi": "#9467bd", "Thriller": "#8c564b", "Adventure": "#ff7f0e",
                 "Action": "#1f77b4", "Comedy": "#e7ba52", "Drama": "#2ca02c", "Other": "#bbbbbb"}
SCORE_NOTES = {
    "neighbours kept": "share of each film's 10 nearest films in the full space that stay among its 10 nearest on "
                       "the map (1 = every neighbourhood survives; chance is about 10 / films).",
    "distance rank": "Spearman correlation of pairwise distances in the full space and on the map, over a sample "
                     "of film pairs (1 = far stays far and near stays near).",
    "profit smoothness": "Spearman correlation of a film's ln(multiple) with the mean of its 10 nearest films on "
                         "the map. Compare it with full-space smoothness, the same measure in the full space: a "
                         "map scoring higher makes profit look more clustered than the vectors are.",
    "net smoothness": "the same as profit smoothness, for the multiple net of budget, year and genre.",
}


def nearest(d: pd.DataFrame, tconst: str, k: int = K) -> pd.DataFrame:
    """The k films nearest to `tconst` on one map (one space and one method), by distance in its 2D x, y.
    Uses the unclipped coordinates; the film itself is left out."""
    d = d.dropna(subset=["x", "y"]).drop_duplicates("tconst").reset_index(drop=True)
    hit = np.flatnonzero(d.tconst.to_numpy() == tconst)
    if not len(hit):
        return d.iloc[:0].assign(distance=pd.Series(dtype=float))
    xy = d[["x", "y"]].to_numpy(float)
    dist = np.hypot(*(xy - xy[hit[0]]).T)
    dist[hit[0]] = np.inf
    idx = np.argsort(dist, kind="stable")[:min(k, len(d) - 1)]
    return d.iloc[idx].assign(distance=dist[idx])


def _range(v: pd.Series) -> list[float]:
    """Axis range cut at the 0.5th and 99.5th percentiles, as in the notebook, so outliers don't squash the cloud."""
    lo, hi = np.nanpercentile(v.to_numpy(float), [0.5, 99.5])
    pad = 0.05 * (hi - lo) or 1.0
    return [lo - pad, hi + pad]


HOVER = "<b>%{customdata[0]}</b> (%{customdata[1]})<br>%{customdata[2]}<br>ln(multiple) %{customdata[3]:.2f}<extra></extra>"


def _custom(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([d.title.fillna(d.tconst), d.year.astype("Int64").astype(str).replace("<NA>", "?"),
                            d.primary_genre.fillna("?"), d.log_mult])


def _panel(fig: go.Figure, d: pd.DataFrame, colour: str, row: int, col: int, first: bool, focus: str | None):
    c = COLOURS[colour]
    if c == "primary_genre":
        g = d.primary_genre.where(d.primary_genre.isin(GENRE_COLOURS), "Other")
        for name, hexc in GENRE_COLOURS.items():
            s = d[g == name]
            if s.empty:
                continue
            fig.add_trace(go.Scattergl(x=s.x, y=s.y, mode="markers", name=f"{name} ({len(s):,})", legendgroup=name,
                                       showlegend=first, marker=dict(color=hexc, size=4, opacity=.7),
                                       customdata=_custom(s), hovertemplate=HOVER), row=row, col=col)
    else:
        has = d[c].notna()
        if (~has).any():
            s = d[~has]
            fig.add_trace(go.Scattergl(x=s.x, y=s.y, mode="markers", name="no value", legendgroup="none",
                                       showlegend=first, marker=dict(color="#cccccc", size=4, opacity=.5),
                                       customdata=_custom(s), hovertemplate=HOVER), row=row, col=col)
        s = d[has]
        fig.add_trace(go.Scattergl(x=s.x, y=s.y, mode="markers", showlegend=False,
                                   marker=dict(color=s[c], coloraxis="coloraxis", size=4, opacity=.7),
                                   customdata=_custom(s), hovertemplate=HOVER), row=row, col=col)
    if focus:
        nb = nearest(d, focus)
        me = d[d.tconst == focus].head(1)
        if not me.empty:
            fig.add_trace(go.Scatter(x=nb.x, y=nb.y, mode="markers", name=f"{K} nearest on the map", legendgroup="nb",
                                     showlegend=first, marker=dict(size=11, color="rgba(0,0,0,0)",
                                                                   line=dict(color="black", width=2)),
                                     customdata=_custom(nb), hovertemplate=HOVER), row=row, col=col)
            fig.add_trace(go.Scatter(x=me.x, y=me.y, mode="markers", name=str(me.title.iloc[0]), legendgroup="me",
                                     showlegend=first, marker=dict(size=16, symbol="star", color="black",
                                                                   line=dict(color="white", width=1)),
                                     customdata=_custom(me), hovertemplate=HOVER), row=row, col=col)
    fig.update_xaxes(range=_range(d.x), row=row, col=col)
    fig.update_yaxes(range=_range(d.y), row=row, col=col)


def _colour_axis(d: pd.DataFrame, colour: str) -> dict:
    c = COLOURS[colour]
    if c == "log_mult":
        return dict(colorscale="Viridis", cmin=-2, cmax=3, colorbar=dict(title="ln(gross / budget)"))
    if c == "log_mult_net":
        v = d[c].dropna().abs()
        m = float(np.percentile(v, 99)) if len(v) else 1.0
        return dict(colorscale="RdBu", cmin=-m, cmax=m, colorbar=dict(title="net ln(multiple)"))
    if c == "year":
        return dict(colorscale="Viridis", colorbar=dict(title="release year"))
    return dict(showscale=False)


def _film_label(films: pd.DataFrame) -> dict[str, str]:
    y = films.year.astype("Int64").astype(str).replace("<NA>", "?")
    return dict(zip(films.tconst, films.title.fillna(films.tconst) + " (" + y + ")"))


def notebook_markdown(path, heading: str | None = None) -> str:
    """The markdown cell of a notebook that starts with `heading`, or with heading=None the **Takeaway.**
    paragraph of its first cell. Empty if the notebook or the cell is missing."""
    try:
        cells = json.loads(path.read_text())["cells"]
    except (OSError, ValueError, KeyError):
        return ""
    texts = ["".join(c["source"]) for c in cells if c.get("cell_type") == "markdown"]
    if heading is None:
        for t in texts:
            for para in t.split("\n\n"):
                if para.startswith("**Takeaway.**"):
                    return para
        return ""
    return next((t for t in texts if t.startswith(heading)), "")


def _scores_file():
    return SCORES if SCORES.exists() else SCORES_COMMITTED if SCORES_COMMITTED.exists() else None


def maps():
    st.title("Embedding maps")
    st.caption("notebooks/embedding_maps.ipynb: each vector space the project builds (screenplays, summaries, plots, "
               "reviews, tags, genres, cast and crew, sentiment arcs) drawn in 2D four ways. Hover a dot for the film. "
               "The checks of these maps are on the Embedding map checks page.")
    st.markdown(md(notebook_markdown(NB_MAPS)))
    if not MAPS.exists():
        missing(f"No embedding maps yet. Run `notebooks/embedding_maps.ipynb`; it writes `{MAPS.name}` to `{MAPS.parent}`. "
                "The scores below are the committed ones.")
        _scores(None)
        _reading()
        return
    allm = table(MAPS)
    allm = allm.assign(tconst=allm.tconst.astype(str))
    spaces = list(dict.fromkeys(allm.space.dropna()))
    if not spaces:
        missing("The embedding maps file has no rows. Run `notebooks/embedding_maps.ipynb` again.")
        return

    a, b, c = st.columns([2, 3, 2])
    space = a.selectbox("Space", spaces, key="maps_space")
    d = allm[allm.space == space].dropna(subset=["x", "y"])
    methods = [m for m in METHODS if m in set(d.method)] + [m for m in dict.fromkeys(d.method) if m not in METHODS]
    method = b.radio("Method", methods + [ALL], horizontal=True, key="maps_method")
    colour = c.selectbox("Colour by", list(COLOURS), key="maps_colour")

    films = d.drop_duplicates("tconst").sort_values(["title", "year"])
    labels = _film_label(films)
    focus = st.selectbox("Find a film (highlights it and its 10 nearest films on the map)", [None] + list(labels),
                         format_func=lambda t: "none" if t is None else labels.get(t, t), key="maps_focus")
    if focus not in labels:
        focus = None

    shown = methods if method == ALL else [method]
    grid = len(shown) > 1
    rows, cols = ((len(shown) + 1) // 2, 2) if grid else (1, 1)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=shown if grid else None,
                        horizontal_spacing=.04, vertical_spacing=.06)
    for i, m in enumerate(shown):
        _panel(fig, d[d.method == m], colour, i // cols + 1, i % cols + 1, i == 0, focus)
    fig.update_xaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False, title=None)
    fig.update_yaxes(showticklabels=False, ticks="", showgrid=False, zeroline=False, title=None)
    fig.update_layout(template="simple_white", height=900 if grid else 640, coloraxis=_colour_axis(d, colour),
                      legend=dict(itemsizing="constant"), margin=dict(t=40, b=10),
                      title=None if grid else f"{space} · {method}")
    st.plotly_chart(fig, width="stretch", key="maps_fig")
    st.caption(f"{films.shape[0]:,} films. Axes have no units and are cut at the 0.5th and 99.5th percentiles, so a "
               "few outliers may sit off the edge. On t-SNE and UMAP the size of a cluster and the gaps between "
               "clusters mean little; only which films sit together does. PCA and Isomap keep more of the global shape.")

    if focus:
        st.subheader(f"Nearest films to {labels[focus]}")
        tabs = st.tabs(shown) if grid else [st.container()]
        for t, m in zip(tabs, shown):
            nb = nearest(d[d.method == m], focus)
            with t:
                if nb.empty:
                    st.caption(f"Not on the {m} map.")
                    continue
                st.dataframe(nb[["title", "year", "log_mult"]].rename(columns={"log_mult": "ln(multiple)"})
                             .round({"ln(multiple)": 3}), width="stretch", hide_index=True, key=f"maps_nb_{m}")
        st.caption("Nearest in the 2D map shown, not in the full space. The map can bring together films that are "
                   "far apart in the full space.")
        if st.button("Open this film in the node explorer", key="maps_open"):
            nav.open_node(focus)

    _scores(space)
    _reading()


def _scores(space: str | None):
    """The scores of one space, or of every space when space is None."""
    st.subheader("How well the maps keep the space")
    f = _scores_file()
    if f is None:
        missing(f"No scores yet. Run `notebooks/embedding_maps.ipynb`; it writes `{SCORES.name}`.")
        return
    sc = table(f)
    if space is not None:
        sc = sc[sc.space == space].drop(columns=["space"])
    if sc.empty:
        st.caption("No scores saved for this space.")
        return
    st.dataframe(sc.round(3), width="stretch", hide_index=True, key="maps_scores")
    for name, note in SCORE_NOTES.items():
        if name in sc.columns:
            st.caption(md(f"**{name}**: {note}"))


def _reading():
    text = notebook_markdown(NB_MAPS, "## 10. What the maps show")
    if text:
        with st.expander("What the maps show and what they hide (the notebook's reading)", expanded=False):
            st.markdown(md(text.split("\n", 1)[1]))


# ---------------------------------------------------------------- checks page

def _committed(name: str) -> pd.DataFrame | None:
    p = COMMITTED / name
    return table(p) if p.exists() else None


def _section(heading: str):
    """A notebook section's intro, without its heading line."""
    text = notebook_markdown(NB_CHECKS, heading)
    if text:
        st.markdown(md(text.split("\n", 1)[1].strip()))


def checks():
    st.title("Embedding map checks")
    st.caption("notebooks/embedding_map_checks.ipynb: how far the embedding maps and spaces can be trusted. Every "
               "number is read from analysis/embedding_maps/, which the notebook writes.")
    st.markdown(md(notebook_markdown(NB_CHECKS)))
    if _committed("stability.csv") is None:
        missing("No check results yet. Run `notebooks/embedding_maps.ipynb`, then `notebooks/embedding_map_checks.ipynb`.")
        return
    t = st.tabs(["Stability", "Spaces agree?", "Clusters", "Release year", "Neighbours by hand", "Reading"])

    with t[0]:
        _section("## 1. Stability")
        s = _committed("stability.csv")
        s = s.assign(redraw_kind=np.where(s.redraw.str.startswith("seed"), "other seed", "other setting"))
        g = (s.groupby(["space", "method", "redraw_kind"], sort=False)[["same neighbours", "same islands"]].mean()
             .reset_index())
        g["run"] = g.method + ", " + g.redraw_kind
        long = g.melt(id_vars=["space", "run"], value_vars=["same neighbours", "same islands"], var_name="score")
        fig = px.bar(long, x="space", y="value", color="run", barmode="group", facet_row="score", height=640,
                     template="simple_white", labels={"value": "", "space": ""},
                     color_discrete_sequence=["#c44e52", "#e8a0a2", "#8172b2", "#c3b9dd"])
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, width="stretch", key="mc_stab_fig")
        show_df(s.drop(columns="redraw_kind").round(3), key="mc_stab")

    with t[1]:
        _section("## 2. Do the spaces agree")
        a = _committed("space_agreement.csv").set_index(_committed("space_agreement.csv").columns[0])
        fig = px.imshow(a.round(3), text_auto=".3f", color_continuous_scale="Viridis", zmin=0, zmax=0.3,
                        aspect="auto", height=620, labels={"color": "neighbours shared"})
        fig.update_layout(template="simple_white", xaxis_title=None, yaxis_title=None)
        st.plotly_chart(fig, width="stretch", key="mc_agree_fig")
        st.caption("Colour capped at 0.3. Tag relevance and tags without reception share 978 of 1,084 columns, so "
                   "their pair is near 1 by construction.")

    with t[2]:
        _section("## 3. Clusters")
        c = _committed("clusters.csv")
        fig = go.Figure()
        for col, name, colour in (("log_mult", "raw multiple", "#4c72b0"),
                                  ("log_mult_net", "net of budget, year and genre", "#dd8452")):
            fig.add_bar(x=c.space, y=c[f"{col} eta2"], name=name, marker_color=colour)
            fig.add_scatter(x=c.space, y=c[f"{col} null 97.5%"], mode="markers", name=f"shuffle 97.5%, {name}",
                            marker=dict(symbol="line-ew-open", size=26, color="black", line=dict(width=2)))
        fig.update_layout(barmode="group", template="simple_white", height=440,
                          yaxis_title="share of variation in ln(multiple)", legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig, width="stretch", key="mc_clusters_fig")
        show_df(c.round(4), key="mc_clusters")
        desc = _committed("cluster_descriptions.csv")
        if desc is not None:
            sp = st.selectbox("Clusters of one space", list(dict.fromkeys(desc.space)), key="mc_desc_space")
            show_df(desc[desc.space == sp].drop(columns="space").sort_values("median multiple", ascending=False)
                    .round(3), key="mc_desc")

    with t[3]:
        _section("## 4. Release year")
        y = _committed("year_smoothness.csv")
        long = y.melt(id_vars="space", var_name="where", value_name="year smoothness")
        fig = px.bar(long, x="space", y="year smoothness", color="where", barmode="group", height=420,
                     template="simple_white", labels={"space": ""})
        st.plotly_chart(fig, width="stretch", key="mc_year_fig")
        show_df(y.round(3), key="mc_year")

    with t[4]:
        _section("## 5. Neighbours by hand")
        sheet = _committed("hand_check_sheet.csv")
        if sheet is None:
            missing("No sheet yet. Run `notebooks/embedding_map_checks.ipynb`.")
        else:
            judged = sheet.similar.notna().sum() if "similar" in sheet else 0
            st.metric("Rows judged", f"{judged} of {len(sheet)}")
            key = _committed("hand_check_key.csv")
            if judged and key is not None:
                j = sheet.merge(key, on="row").dropna(subset=["similar"])
                score = j.groupby(["space", "neighbour"]).similar.mean().unstack("neighbour")
                score.columns = ["random film" if not c else "nearest film" for c in score.columns]
                st.caption("Share judged similar: nearest films against the random film mixed in.")
                st.dataframe(score.round(2), width="stretch")
            st.caption("The sheet, without the key. Judge it in the CSV, not here, and don't open the key first.")
            show_df(sheet, key="mc_sheet")

    with t[5]:
        text = notebook_markdown(NB_CHECKS, "## 6. What the checks show")
        st.markdown(md(text.split("\n", 1)[1]) if text else "No reading in the notebook yet.")
