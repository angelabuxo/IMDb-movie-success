"""Whole graph, node explorer and screenplay reader.

Everything is read from data/raw on this machine and nothing is written. Screenplays are copyrighted
(MovieSum, CC BY-NC 4.0 for the dataset, the scripts themselves under copyright): they are shown here for
checking only and never leave this app.
"""
from __future__ import annotations

import html
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sp
import streamlit as st

from . import nav
from .common import DATA, FEATURES, OUTPUT_DIRS, REPO, md, missing, mtime, show_df, table

GRAPH = FEATURES / "graph"
NODES, EDGES = GRAPH / "nodes.csv", GRAPH / "edges.csv"
MOVIESUM = [DATA / "scripts" / "moviesum" / f"{s}.jsonl" for s in ("train", "val", "test")]
PEOPLE_ROLES = {"director", "writer", "producer", "composer", "cinematographer", "editor"}
COMPANY_ROLES = {"studio", "distributor"}
TYPE_COLOUR = {"movie": "#0072B2", "person": "#E69F00", "company": "#009E73"}
ENTITY_MODES = ("People who worked together", "Companies that co-produce")   # nodes are people or companies
QUAL = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#999999",
        "#882255", "#44AA99", "#117733", "#332288", "#AA4499", "#DDCC77", "#88CCEE", "#661100"]


def _need_graph() -> bool:
    if not NODES.exists() or not EDGES.exists():
        st.warning(f"Needs the movie graph in `{GRAPH}` (see the README: build_graph.py).")
        return False
    return True


# ---------------------------------------------------------------- graph data

@st.cache_resource(show_spinner="Indexing the movie graph…")
def graph_data(stamp: float) -> dict:
    n = pd.read_csv(NODES, low_memory=False)
    for c in ["year", "votes", "rating", "budget_usd", "gross_usd", "multiple", "rt", "mc", "birth_year", "death_year"]:
        n[c] = pd.to_numeric(n[c], errors="coerce")
    n["clean"] = n["clean"].astype(str).isin(["1", "1.0", "True"])
    n["log_multiple"] = np.log(n.multiple.where(n.multiple > 0))
    e = pd.read_csv(EDGES, low_memory=False)
    e["ordering"] = pd.to_numeric(e.ordering, errors="coerce")
    e["key_person"] = e.role.isin(PEOPLE_ROLES) | ((e.role == "actor") & (e.ordering <= 3))
    e["company"] = e.role.isin(COMPANY_ROLES)
    deg = pd.concat([e.source, e.target]).value_counts()
    n["degree"] = n.id.map(deg).fillna(0).astype(int)
    return {"nodes": n, "by_id": n.set_index("id", drop=False), "edges": e,
            "label": dict(zip(n.id, n.label.astype(str)))}


def gd() -> dict:
    return graph_data(mtime(NODES) + mtime(EDGES))


@st.cache_resource(show_spinner="Building the film-by-film links…")
def incidence(stamp: float, kind: str):
    """Film x entity incidence, each pair once. kind: 'people' (key people), 'companies' (studios and
    distributors) or 'studios' (production companies only: the infobox 'studios' field)."""
    g = graph_data(stamp)
    films = g["nodes"].loc[g["nodes"].type == "movie", "id"].to_numpy()
    fi = {t: k for k, t in enumerate(films)}
    ed = g["edges"]
    e = ed[ed.role == "studio"] if kind == "studios" else ed[ed["key_person" if kind == "people" else "company"]]
    e = e[e.target.isin(fi)].drop_duplicates(["source", "target"])
    codes, ents = pd.factorize(e.source)
    B = sp.csr_matrix((np.ones(len(e), dtype=np.float32), (e.target.map(fi).to_numpy(), codes)),
                      shape=(len(films), len(ents)))
    return films, np.asarray(ents), B


def _layout(edges: np.ndarray, weights: np.ndarray, n: int, seed: int = 2805) -> np.ndarray:
    G = ig.Graph(n=n, edges=edges.tolist())
    if n > 1500:
        return np.array(G.layout_drl(weights=weights.tolist()).coords)
    return np.array(G.layout_fruchterman_reingold(weights=weights.tolist(), niter=600, seed=None).coords)


@st.cache_data(show_spinner="Laying out the graph (cached after the first time)…", max_entries=16)
def build_view(stamp: float, mode: str, tiers: tuple, clean_only: bool, years: tuple, genre: str,
               min_w: int, max_nodes: int, max_edges: int, min_films: int) -> dict:
    g = graph_data(stamp)
    by = g["by_id"]
    kind = {"Films sharing a company": "companies", "Companies that co-produce": "studios"}.get(mode, "people")
    films, ents, B = incidence(stamp, kind)
    f = by.loc[films]
    keep = f.tier.isin(tiers) & f.year.between(*years)
    if clean_only:
        keep &= f.clean
    if genre != "any":
        keep &= f.genres.fillna("").str.split(",").apply(lambda s: genre in s)
    keep = keep.to_numpy()
    node_lm = node_n = None
    if mode in ENTITY_MODES:
        Bf = B[keep]
        nfilms = np.asarray(Bf.sum(0)).ravel()
        sel = np.flatnonzero(nfilms >= min_films)
        A = (Bf[:, sel].T @ Bf[:, sel]).tocsr()
        ids = ents[sel]
        # each person's or company's films in the selection: how many, and their median ln(gross / budget)
        inc = Bf[:, sel].tocoo()
        lm = f.log_multiple.to_numpy()[keep][inc.row]
        node_lm = pd.Series(lm).groupby(inc.col).median().reindex(range(len(sel))).to_numpy()
        node_n = nfilms[sel].astype(int)
    else:
        idx = np.flatnonzero(keep)
        A = (B[idx] @ B[idx].T).tocsr()
        ids = films[idx]
    A.setdiag(0)
    A.eliminate_zeros()
    A = sp.triu(A, k=1).tocoo()
    m = A.data >= min_w
    r, c, w = A.row[m], A.col[m], A.data[m]
    # keep the max_nodes nodes with the largest weighted degree, and drop isolates
    wdeg = np.bincount(r, w, len(ids)) + np.bincount(c, w, len(ids))
    top = np.argsort(-wdeg)[:max_nodes]
    top = top[wdeg[top] > 0]
    pos_of = -np.ones(len(ids), dtype=int)
    pos_of[top] = np.arange(len(top))
    m = (pos_of[r] >= 0) & (pos_of[c] >= 0)
    r, c, w = pos_of[r[m]], pos_of[c[m]], w[m]
    nid = ids[top]
    if len(nid) == 0:
        return {"ids": [], "r": [], "c": [], "w": [], "xy": np.zeros((0, 2)), "comm": [], "modularity": 0.0,
                "node_lm": None, "node_n": None}
    edges = np.column_stack([r, c])
    xy = _layout(edges, w, len(nid))
    G = ig.Graph(n=len(nid), edges=edges.tolist())
    part = G.community_multilevel(weights=w.tolist())
    order = np.argsort(-w)[:max_edges]
    return {"ids": nid.tolist(), "r": r[order], "c": c[order], "w": w[order], "n_links": int(len(w)),
            "xy": xy, "comm": np.array(part.membership), "modularity": float(part.modularity),
            "deg": np.bincount(r, w, len(nid)) + np.bincount(c, w, len(nid)),
            "node_lm": None if node_lm is None else node_lm[top], "node_n": None if node_n is None else node_n[top]}


def _edge_trace(xy, r, c, width=0.5, opacity=0.25, colour="#888888"):
    x = np.column_stack([xy[r, 0], xy[c, 0], np.full(len(r), np.nan)]).ravel()
    y = np.column_stack([xy[r, 1], xy[c, 1], np.full(len(r), np.nan)]).ravel()
    return go.Scattergl(x=x, y=y, mode="lines", line=dict(width=width, color=colour), opacity=opacity,
                        hoverinfo="skip", showlegend=False)


def _selected_ids(event) -> list[str]:
    try:
        pts = event.selection["points"] if hasattr(event, "selection") else event["selection"]["points"]
    except (KeyError, TypeError, AttributeError):
        return []
    out = []
    for p in pts:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd:
            out.append(str(cd))
    return list(dict.fromkeys(out))


def _node_summary(nid: str) -> pd.DataFrame:
    by = gd()["by_id"]
    if nid not in by.index:
        return pd.DataFrame()
    r = by.loc[nid]
    keep = [c for c in ["id", "type", "label", "year", "genres", "budget_usd", "gross_usd", "multiple", "clean", "tier",
                        "votes", "rating", "roles", "birth_year", "death_year", "degree"] if pd.notna(r.get(c))]
    return r[keep].to_frame("value").T


# ---------------------------------------------------------------- links out and the film overview

LINKS = REPO / "data" / "movie_links.csv"


@st.cache_data(show_spinner=False)
def _wiki_links(stamp: float) -> dict:
    if not LINKS.exists():
        return {}
    d = pd.read_csv(LINKS, usecols=["imdb_id", "wikipedia_url", "wikidata_url"], dtype=str).fillna("")
    return d.drop_duplicates("imdb_id").set_index("imdb_id").to_dict("index")


@st.cache_data(show_spinner=False)
def _tagged_films(stamp: float) -> frozenset:
    """Films with tag scores: Tag Genome, or predicted from the plot summary when that file is here."""
    out = set()
    for name in ("tag_relevance_ids.csv", "plot_tag_relevance_ids.csv"):
        p = FEATURES / name
        if p.exists():
            out |= set(pd.read_csv(p)["tconst"])
    return frozenset(out)


def external_links(nid: str) -> dict[str, str]:
    """Where a node lives outside the app: IMDb from its ID, Wikipedia and Wikidata from data/movie_links.csv."""
    nid, out = str(nid), {}
    if re.fullmatch(r"tt\d+", nid):
        out["IMDb"] = f"https://www.imdb.com/title/{nid}/"
        row = _wiki_links(mtime(LINKS)).get(nid, {})
        for name, col in (("Wikipedia", "wikipedia_url"), ("Wikidata", "wikidata_url")):
            if str(row.get(col, "")).startswith("http"):
                out[name] = row[col]
    elif re.fullmatch(r"nm\d+", nid):
        out["IMDb"] = f"https://www.imdb.com/name/{nid}/"
    elif re.fullmatch(r"company:Q\d+", nid):
        out["Wikidata"] = "https://www.wikidata.org/wiki/" + nid.split(":")[1]
    return out


def link_buttons(nid: str, columns=None):
    """One button per outside link, each opening in a new browser tab."""
    links = external_links(nid)
    cols = columns if columns is not None else st.columns(max(len(links), 1) + 3)[:len(links)]
    for c, (name, url) in zip(cols, links.items()):
        c.link_button(f"{name} ↗", url, width="stretch")


def _names(e: pd.DataFrame, labels: dict, role: str, k: int) -> str:
    d = e[e.role == role].sort_values("ordering")
    return ", ".join(labels.get(x, x) for x in d.source.head(k))


def film_overview(tconst: str, key: str, note: str | None = None, fallback: dict | None = None, tags_button: bool = False):
    """A film at a glance: money, ratings, who made it, and buttons to its full page here, IMDb and Wikipedia.
    `fallback` gives title, year and genres when the movie graph isn't on this machine; `tags_button` adds a
    button that opens the film on the Tags and genres page."""
    r, e, labels = None, None, {}
    if NODES.exists() and EDGES.exists():
        g = gd()
        if tconst in g["by_id"].index:
            r, labels = g["by_id"].loc[tconst], g["label"]
            e = g["edges"][g["edges"].target == tconst]
    f = fallback or {}
    title = r.label if r is not None else f.get("title", tconst)
    year = r.year if r is not None else f.get("year")
    genres = r.genres if r is not None else f.get("genres")
    with st.container(border=True):
        st.markdown(md(f"#### {title}" + (f" ({int(year)})" if pd.notna(year) else "")))
        st.caption(" · ".join(x for x in [str(genres).replace(",", ", ") if pd.notna(genres) else "", f"`{tconst}`"] if x))
        if r is not None:
            c = st.columns(5)
            c[0].metric("Budget", f"${r.budget_usd / 1e6:,.1f}M" if pd.notna(r.budget_usd) else "—")
            c[1].metric("Gross", f"${r.gross_usd / 1e6:,.1f}M" if pd.notna(r.gross_usd) else "—")
            c[2].metric("Gross ÷ budget", f"{r.multiple:,.2f}" if pd.notna(r.multiple) else "—")
            c[3].metric("IMDb rating", f"{r.rating:.1f}" if pd.notna(r.rating) else "—",
                        help=f"{int(r.votes):,} votes" if pd.notna(r.votes) else None)
            c[4].metric("Critics", " · ".join(x for x in [f"RT {int(r.rt)}%" if pd.notna(r.rt) else "",
                                                          f"MC {int(r.mc)}" if pd.notna(r.mc) else ""] if x) or "—")
            who = [("Directed by", _names(e, labels, "director", 3)), ("Starring", _names(e, labels, "actor", 4)),
                   ("Studio", _names(e, labels, "studio", 3))]
            line = " · ".join(f"**{k}** {v}" for k, v in who if v)
            if line:
                st.markdown(md(line))
        if note:
            st.markdown(md(note))
        links = external_links(tconst)
        cols = st.columns(len(links) + 2 + (2 if len(links) < 3 else 1))
        k = 0
        if r is not None:
            if cols[k].button("Full film page", key=f"open_{key}", type="primary", width="stretch",
                              help="Everything on this film in the node explorer: credits, linked films, all fields."):
                nav.open_node(tconst)
            k += 1
        if tags_button:
            if cols[k].button("Its tags", key=f"tags_{key}", width="stretch",
                              help="This film's genre shares and tags on the One film view."):
                nav.open_tags(tconst)
            k += 1
        link_buttons(tconst, cols[k:k + len(links)])


# ---------------------------------------------------------------- whole graph

def whole_graph():
    st.title("Whole graph")
    if not _need_graph():
        return
    g = gd()
    n, e = g["nodes"], g["edges"]
    counts = {"movie": "Movies", "person": "People", "company": "Companies"}
    st.markdown(f"**{len(n):,} nodes** (" + ", ".join(f"{int((n.type == t).sum()):,} {lab.lower()}" for t, lab in counts.items())
                + f") and **{len(e):,} credits**.")
    st.caption("The full graph has about 110,000 nodes, too many to draw at once. Below it is projected: films linked "
               "when they share people (or companies), or people (or production companies) linked when they made films "
               "together. Filter, then "
               "**click a node, or drag a box around several** (lasso is in the chart's toolbar), to see its details.")

    with st.sidebar:
        st.subheader("What to draw")
        mode = st.radio("Network", ["Films sharing people", "Films sharing a company", "People who worked together",
                                    "Companies that co-produce"], key="graph_mode")
        tiers = tuple(st.multiselect("Vote tier", ["100k+", "10k-100k"], default=["100k+", "10k-100k"]))
        clean_only = st.toggle("Clean films only", False)
        years = st.slider("Release year", 1915, 2026, (1990, 2026))
        genres = sorted({x for s in n.genres.dropna() for x in s.split(",") if x})
        genre = st.selectbox("Genre", ["any"] + genres)
        min_w = st.slider("Minimum shared people / companies / films on a link", 1, 6, 2 if mode != "Films sharing a company" else 1)
        min_films = (st.slider("Minimum films in the selection (per person or company)", 1, 20,
                               5 if mode == "People who worked together" else 3) if mode in ENTITY_MODES else 1)
        max_nodes = st.slider("Max nodes (most connected first)", 200, 8000, 2500, step=100)
        max_edges = st.slider("Max links drawn (strongest first)", 0, 40000, 8000, step=1000)
        colour = st.selectbox("Colour by", ["community", "median log(gross / budget) of their films"] if mode in ENTITY_MODES
                              else ["community", "log(gross / budget)", "release year", "first genre"])
    if not tiers:
        st.info("Choose at least one tier.")
        return
    v = build_view(mtime(NODES) + mtime(EDGES), mode, tiers, clean_only, years, genre, min_w, max_nodes, max_edges, min_films)
    if not v["ids"]:
        st.info("No links in this selection. Lower the minimum shared count or widen the filters.")
        return
    ids = np.array(v["ids"])
    xy, comm = v["xy"], v["comm"]
    by = g["by_id"].loc[ids]
    st.caption(f"{len(ids):,} nodes, {v['n_links']:,} links ({len(v['r']):,} drawn), {comm.max() + 1} communities, "
               f"modularity {v['modularity']:.3f}")
    entity = mode in ENTITY_MODES
    node_lm = v["node_lm"] if entity else by.log_multiple.to_numpy()
    if mode == "Companies that co-produce":
        st.caption("Production companies (the infobox *studios* field; distributors left out), linked by how many films "
                   "they co-produced in the selection. Colour by the median log(gross / budget) of each company's films to "
                   "see whether co-producing circles share a return level.")

    fig = go.Figure()
    if len(v["r"]):
        fig.add_trace(_edge_trace(xy, v["r"], v["c"], opacity=.18 if len(v["r"]) > 3000 else .35))
    size = 4 + 14 * np.sqrt(v["deg"] / v["deg"].max())
    if entity:
        hover = [f"<b>{html.escape(str(l))}</b><br>{t} · {k_n:,} films in the selection"
                 + (f"<br>median ln(gross/budget) {lm:.2f}" if lm == lm else "") + f"<br>community {k}"
                 for l, t, k_n, lm, k in zip(by.label, by.type, v["node_n"], node_lm, comm)]
    else:
        hover = [f"<b>{html.escape(str(l))}</b><br>{t}" + (f" · {int(y)}" if y == y else "") +
                 (f"<br>gross/budget {m:.2f}" if m == m else "") + f"<br>community {k}"
                 for l, t, y, m, k in zip(by.label, by.type, by.year, by.multiple, comm)]
    if colour == "community":
        big = pd.Series(comm).value_counts().index[:len(QUAL)]
        col = [QUAL[list(big).index(k)] if k in big else "#cccccc" for k in comm]
        marker = dict(color=col)
    elif colour in ("log(gross / budget)", "median log(gross / budget) of their films"):
        marker = dict(color=np.clip(node_lm, -3, 3), colorscale="RdBu", cmid=0, showscale=True,
                      colorbar=dict(title="ln multiple"))
    elif colour == "release year":
        marker = dict(color=by.year.to_numpy(), colorscale="Viridis", showscale=True, colorbar=dict(title="year"))
    else:
        fg = by.genres.fillna("Unknown").str.split(",").str[0]
        top = fg.value_counts().index[:len(QUAL)]
        marker = dict(color=[QUAL[list(top).index(x)] if x in top else "#cccccc" for x in fg])
    fig.add_trace(go.Scattergl(x=xy[:, 0], y=xy[:, 1], mode="markers", customdata=ids, text=hover,
                               hovertemplate="%{text}<extra></extra>",
                               marker=dict(size=size, line=dict(width=0.3, color="white"), **marker), showlegend=False))
    fig.update_layout(clickmode="event+select", template="simple_white", height=760, margin=dict(l=0, r=0, t=10, b=0), dragmode="select",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"),
                            key="whole_graph_plot")
    sel = _selected_ids(event)
    jump = st.selectbox("Or find a node in this drawing", sorted(ids, key=lambda x: g["label"].get(x, x).lower()), index=None,
                        placeholder="Type a title or name…", format_func=lambda x: g["label"].get(x, x), key="graph_jump")
    if jump and jump not in sel:
        sel = [jump] + sel

    left, right = st.columns([3, 2])
    with left:
        if sel:
            s = g["by_id"].loc[sel]
            st.markdown(f"**{len(sel):,} selected**")
            if entity:   # people and companies: their films in the selection, not film fields
                pos = {x: k for k, x in enumerate(ids)}
                s = s.assign(films_in_selection=[v["node_n"][pos[x]] for x in sel],
                             median_log_multiple=[round(float(node_lm[pos[x]]), 3) for x in sel])
                cols = ["id", "label", "type", "films_in_selection", "median_log_multiple", "degree"]
            else:
                cols = [c for c in ["id", "label", "type", "year", "genres", "multiple", "clean", "degree"] if c in s]
            pick = st.dataframe(s[cols], width="stretch", hide_index=True, on_select="rerun",
                                selection_mode="single-row", key="sel_table")
            rows = pick.selection.rows if hasattr(pick, "selection") else []
            target = s.id.iloc[rows[0]] if rows else sel[0]
            st.dataframe(_node_summary(target), width="stretch", hide_index=True)
            b1, b2 = st.columns(2)
            if b1.button(f"Open “{g['label'].get(target, target)}” in the node explorer", type="primary"):
                nav.open_node(target)
            if target.startswith("tt") and b2.button("Read its screenplay"):
                nav.open_script(target)
        else:
            st.info("Click a node, or drag a box or lasso around several, to see them here.")
    with right:
        st.markdown("**Communities** (Louvain on the drawn network)")
        cdf = pd.DataFrame({"community": comm, "id": ids, "label": by.label.values, "deg": v["deg"],
                            "log_multiple": node_lm})
        summ = (cdf.sort_values("deg", ascending=False).groupby("community")
                .agg(size=("id", "size"), median_log_multiple=("log_multiple", "median"),
                     biggest=("label", lambda s: ", ".join(map(str, s.head(4)))))
                .sort_values("size", ascending=False).reset_index())
        st.dataframe(summ.round(3), width="stretch", hide_index=True, height=360)

    with st.expander("Degree distribution of the full bipartite graph"):
        d = n[n.degree > 0]
        fig = go.Figure()
        for t, colr in TYPE_COLOUR.items():
            vc = d.loc[d.type == t, "degree"].value_counts().sort_index()
            fig.add_scatter(x=vc.index, y=vc.values, mode="markers", name=t, marker=dict(color=colr, size=5))
        fig.update_layout(clickmode="event+select", template="simple_white", xaxis_type="log", yaxis_type="log", height=380,
                          xaxis_title="credits (degree)", yaxis_title="nodes")
        st.plotly_chart(fig, width="stretch")
        G = ig.Graph.TupleList(e[["source", "target"]].itertuples(index=False), directed=False)
        comps = G.connected_components()
        st.caption(f"{len(comps):,} connected components; the largest holds {max(comps.sizes()):,} of {G.vcount():,} "
                   "connected nodes.")


# ---------------------------------------------------------------- film-only Q2 network

@st.cache_data(show_spinner="Building the film-only network…", max_entries=12)
def movie_network_view(stamp: float, roles: tuple, tiers: tuple, clean_only: bool, years: tuple,
                       genre: str, max_nodes: int, max_edges: int) -> dict:
    """Build the film projection for Q2, retaining the displayed graph's centralities."""
    g = graph_data(stamp)
    n, e = g["nodes"], g["edges"]
    films = n[n.type == "movie"].copy()
    keep = films.tier.isin(tiers) & films.year.between(*years)
    if clean_only:
        keep &= films.clean
    if genre != "any":
        keep &= films.genres.fillna("").str.split(",").apply(lambda xs: genre in xs)
    films = films.loc[keep].reset_index(drop=True)
    if films.empty:
        return {"ids": [], "r": [], "c": [], "w": [], "xy": np.zeros((0, 2))}

    role_map = {"director": "director", "main actor": "actor", "producer": "producer", "studio": "studio"}
    selected = e[e.role.isin([role_map[r] for r in roles]) & e.target.isin(set(films.id))].copy()
    selected = selected[(selected.role != "actor") | (selected.ordering == 1)]
    selected["entity"] = selected.role + "|" + selected.source
    selected = selected.drop_duplicates(["target", "entity"])
    fi = {t: k for k, t in enumerate(films.id)}
    entities, codes = np.unique(selected.entity, return_inverse=True)
    B = sp.csr_matrix((np.ones(len(selected), dtype=np.float32),
                       (selected.target.map(fi).to_numpy(), codes)),
                      shape=(len(films), len(entities)))
    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A = sp.triu(A, k=1).tocoo()
    if not len(A.data):
        return {"ids": [], "r": [], "c": [], "w": [], "xy": np.zeros((0, 2))}

    weighted_degree = np.bincount(A.row, A.data, len(films)) + np.bincount(A.col, A.data, len(films))
    top = np.argsort(-weighted_degree)[:max_nodes]
    top = top[weighted_degree[top] > 0]
    pos = -np.ones(len(films), dtype=int)
    pos[top] = np.arange(len(top))
    edge_keep = (pos[A.row] >= 0) & (pos[A.col] >= 0)
    r, c, w = pos[A.row[edge_keep]], pos[A.col[edge_keep]], A.data[edge_keep]
    if not len(r):
        return {"ids": [], "r": [], "c": [], "w": [], "xy": np.zeros((0, 2))}
    order = np.argsort(-w)[:max_edges] if max_edges else np.array([], dtype=int)
    edges = np.column_stack([r, c])
    graph = ig.Graph(n=len(top), edges=edges.tolist())
    xy = _layout(edges, w, len(top))
    community_methods = {
        "Louvain": graph.community_multilevel(weights=w.tolist()),
        "Leiden": graph.community_leiden(objective_function="modularity", weights=w.tolist(),
                                         resolution=1.0, beta=0.01, n_iterations=10),
        "Infomap": graph.community_infomap(edge_weights=w.tolist(), trials=10),
        "Label propagation": graph.community_label_propagation(weights=w.tolist()),
    }
    communities = {}
    method_comparison = []
    for name, partition in community_methods.items():
        membership = np.asarray(partition.membership)
        communities[name] = membership
        method_comparison.append({
            "method": name,
            "communities": int(len(set(membership))),
            "modularity": round(float(graph.modularity(membership, weights=w.tolist())), 4),
        })
    return {
        "ids": films.id.to_numpy()[top].tolist(),
        "r": r[order], "c": c[order], "w": w[order], "xy": xy,
        "degree": np.asarray(graph.degree()), "weighted_degree": np.asarray(graph.strength(weights=w.tolist())),
        "pagerank": np.asarray(graph.pagerank(weights=w.tolist())),
        "links": int(len(r)), "isolates": int(len(films) - len(top)),
        "communities": communities, "community_comparison": method_comparison,
    }


def _community_summary(ids: np.ndarray, by: pd.DataFrame, membership: np.ndarray) -> pd.DataFrame:
    frame = by.loc[ids, ["label", "log_multiple"]].copy()
    frame["community"] = membership
    rows = []
    for community, group in frame.groupby("community"):
        values = pd.to_numeric(group["log_multiple"], errors="coerce").dropna()
        rows.append({
            "community": int(community) + 1,
            "films": int(len(group)),
            "total_ln_multiple": round(float(values.sum()), 3) if len(values) else None,
            "average": round(float(values.mean()), 3) if len(values) else None,
            "median": round(float(values.median()), 3) if len(values) else None,
            "minimum": round(float(values.min()), 3) if len(values) else None,
            "maximum": round(float(values.max()), 3) if len(values) else None,
            "range": round(float(values.max() - values.min()), 3) if len(values) else None,
        })
    return pd.DataFrame(rows).sort_values(["films", "community"], ascending=[False, True])


def _movie_network_detail(tconst: str, centrality: pd.Series) -> None:
    g = gd()
    if tconst not in g["by_id"].index:
        return
    r = g["by_id"].loc[tconst]
    e = g["edges"]
    credits = e[e.target == tconst]
    labels = g["label"]
    who = {
        "Director": _names(credits, labels, "director", 3),
        "Main actor": _names(credits[credits.ordering == 1], labels, "actor", 3),
        "Producer": _names(credits, labels, "producer", 3),
        "Studio": _names(credits, labels, "studio", 3),
    }
    st.subheader(f"{r.label} ({int(r.year) if pd.notna(r.year) else 'year unknown'})")
    st.caption(f"`{tconst}` · {str(r.genres).replace(',', ', ')}")
    cols = st.columns(7)
    cols[0].metric("Budget", f"${r.budget_usd / 1e6:,.1f}M" if pd.notna(r.budget_usd) else "—")
    cols[1].metric("Gross", f"${r.gross_usd / 1e6:,.1f}M" if pd.notna(r.gross_usd) else "—")
    cols[2].metric("ln(gross/budget)", f"{r.log_multiple:+.3f}" if pd.notna(r.log_multiple) else "—")
    cols[3].metric("IMDb rating", f"{r.rating:.1f}" if pd.notna(r.rating) else "—")
    cols[4].metric("Degree", f"{centrality.get('degree', np.nan):,.0f}" if pd.notna(centrality.get("degree")) else "—")
    cols[5].metric("PageRank", f"{centrality.get('pagerank', np.nan):.4f}" if pd.notna(centrality.get("pagerank")) else "—")
    cols[6].metric("Community", f"{int(centrality.get('community', 0)) + 1}"
                   if pd.notna(centrality.get("community")) else "—")
    st.markdown(" · ".join(f"**{key}:** {value}" for key, value in who.items() if value) or "No requested credits found.")
    link_buttons(tconst)


def movie_network():
    """Interactive film-only network for Q2: shared director, main actor, producer, or studio."""
    st.title("Film-only network")
    if not _need_graph():
        return
    g = gd()
    n = g["nodes"]
    st.caption("Q2: films are linked when they share a director, top-billed actor, producer, or studio. "
               "Drag to pan, scroll to zoom, hover for details, and click or lasso films to inspect them.")
    genres = sorted({x for s in n.loc[n.type == "movie", "genres"].dropna() for x in s.split(",") if x})
    with st.sidebar:
        st.subheader("Film network")
        roles = tuple(st.multiselect("Shared credit types", ["director", "main actor", "producer", "studio"],
                                     default=["director", "main actor", "producer", "studio"], key="movie_net_roles"))
        tiers = tuple(st.multiselect("Vote tier", ["100k+", "10k-100k"],
                                     default=["100k+", "10k-100k"], key="movie_net_tiers"))
        clean_only = st.toggle("Clean films only", True, key="movie_net_clean")
        years = st.slider("Release year", 1915, 2026, (1975, 2025), key="movie_net_years")
        genre = st.selectbox("Genre", ["any"] + genres, key="movie_net_genre")
        max_nodes = st.slider("Maximum films to draw", 200, 3000, 1200, step=100, key="movie_net_nodes")
        max_edges = st.slider("Maximum links to draw", 1000, 30000, 10000, step=1000, key="movie_net_edges")
        community_method = st.selectbox("Community method", ["Louvain", "Leiden", "Infomap", "Label propagation"],
                                        key="movie_net_community")
        colour = st.selectbox("Colour by", ["ln(gross/budget)", "release year", "degree", "PageRank", "community"],
                              key="movie_net_colour")
    if not roles or not tiers:
        st.info("Choose at least one credit type and vote tier.")
        return
    v = movie_network_view(mtime(NODES) + mtime(EDGES), roles, tiers, clean_only, years, genre, max_nodes, max_edges)
    if not v["ids"]:
        st.info("No linked films in this selection. Widen the filters.")
        return
    ids = np.asarray(v["ids"])
    by = g["by_id"].loc[ids]
    centrality = pd.DataFrame({"degree": v["degree"], "weighted_degree": v["weighted_degree"],
                               "pagerank": v["pagerank"]}, index=ids)
    membership = v["communities"][community_method]
    centrality["community"] = membership
    st.caption(f"{len(ids):,} films shown, {v['links']:,} links in the filtered network "
               f"({len(v['r']):,} drawn); {v['isolates']:,} filtered films had no link.")
    with st.expander("Communities and log gross/budget summaries"):
        st.caption("Communities use all weighted links inside the displayed 1,200-film cap, not only the strongest "
                   "10,000 links shown. Infomap optimizes a flow description length rather than modularity; "
                   "modularity is shown here only as a comparable diagnostic.")
        st.dataframe(pd.DataFrame(v["community_comparison"]), width="stretch", hide_index=True)
        st.dataframe(_community_summary(ids, by, membership), width="stretch", hide_index=True)

    fig = go.Figure()
    if len(v["r"]):
        fig.add_trace(_edge_trace(v["xy"], v["r"], v["c"], opacity=.2 if len(v["r"]) > 3000 else .35))
    if colour == "ln(gross/budget)":
        colours, colourbar = np.clip(by.log_multiple.to_numpy(), -3, 3), dict(title="ln multiple")
        marker = dict(color=colours, colorscale="RdBu", cmid=0, showscale=True, colorbar=colourbar)
    elif colour == "release year":
        marker = dict(color=by.year.to_numpy(), colorscale="Viridis", showscale=True, colorbar=dict(title="year"))
    elif colour == "degree":
        marker = dict(color=v["degree"], colorscale="Plasma", showscale=True, colorbar=dict(title="degree"))
    elif colour == "PageRank":
        marker = dict(color=v["pagerank"], colorscale="Plasma", showscale=True, colorbar=dict(title="PageRank"))
    else:
        unique = np.unique(membership)
        palette = [QUAL[i % len(QUAL)] for i in range(len(unique))]
        colour_of = dict(zip(unique, palette))
        marker = dict(color=[colour_of[x] for x in membership])
    hover = [
        f"<b>{html.escape(str(title))}</b><br>{int(year) if pd.notna(year) else 'year unknown'} · {tconst}"
        f"<br>ln(gross/budget): {lm:+.3f}" if pd.notna(lm) else
        f"<b>{html.escape(str(title))}</b><br>{int(year) if pd.notna(year) else 'year unknown'} · {tconst}<br>return unavailable"
        for title, year, tconst, lm in zip(by.label, by.year, ids, by.log_multiple)
    ]
    hover = [text + f"<br>degree: {degree} · weighted degree: {weight:.0f}<br>PageRank: {rank:.4f}"
             f"<br>{community_method} community: {int(community) + 1}"
             for text, degree, weight, rank, community in zip(
                 hover, v["degree"], v["weighted_degree"], v["pagerank"], membership)]
    fig.add_trace(go.Scattergl(x=v["xy"][:, 0], y=v["xy"][:, 1], mode="markers", customdata=ids, text=hover,
                               hovertemplate="%{text}<extra></extra>",
                               marker=dict(size=5 + 13 * np.sqrt(v["weighted_degree"] / max(v["weighted_degree"].max(), 1)),
                                           line=dict(width=.3, color="white"), **marker), showlegend=False))
    fig.update_layout(clickmode="event+select", template="simple_white", height=780,
                      margin=dict(l=0, r=0, t=10, b=0), dragmode="select",
                      xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
    event = st.plotly_chart(fig, width="stretch", on_select="rerun",
                            selection_mode=("points", "box", "lasso"), key="movie_network_plot")
    selected = _selected_ids(event)
    choices = sorted(ids, key=lambda x: str(g["label"].get(x, x)).lower())
    jump = st.selectbox("Or find a film", choices, index=None, placeholder="Type a title or IMDb ID…",
                        format_func=lambda x: g["label"].get(x, x), key="movie_net_jump")
    if jump and jump not in selected:
        selected = [jump] + selected
    if selected:
        target = selected[0]
        _movie_network_detail(target, centrality.loc[target])
        if len(selected) > 1:
            st.markdown(f"**{len(selected):,} films selected**")
            st.dataframe(by.loc[selected, ["label", "year", "genres", "budget_usd", "gross_usd", "log_multiple"]]
                         .join(centrality).rename(columns={"label": "title"}), width="stretch", hide_index=True)
    else:
        st.info("Click a film, or drag a box/lasso around films, to see its credits, money and centralities.")


# ---------------------------------------------------------------- node explorer

def _ego(nid: str, hops: int, cap: int):
    g = gd()
    e = g["edges"]
    first = e[(e.source == nid) | (e.target == nid)]
    nb = set(first.source) | set(first.target)
    links = first
    if hops == 2:
        second = e[e.source.isin(nb) | e.target.isin(nb)]
        far = pd.concat([second.source, second.target])
        far = far[~far.isin(nb)].value_counts().head(cap)
        second = second[second.source.isin(set(far.index) | nb) & second.target.isin(set(far.index) | nb)]
        links = pd.concat([first, second]).drop_duplicates(["source", "target", "role"])
    pairs = links.groupby(["source", "target"]).role.agg(lambda s: ", ".join(sorted(set(s)))).reset_index()
    ids = pd.unique(pd.concat([pairs.source, pairs.target]))
    return ids, pairs


def node():
    st.title("Node explorer")
    if not _need_graph():
        return
    g = gd()
    n, e, by = g["nodes"], g["edges"], g["by_id"]
    hist = st.session_state.setdefault("node_history", [])
    cur = st.session_state.get("node_id") or "tt1375666"
    if cur not in by.index:
        cur = n.loc[n.type == "movie"].sort_values("votes", ascending=False).id.iloc[0]

    with st.sidebar:
        st.subheader("Find a node")
        q = st.text_input("Name, title or ID", placeholder="e.g. Christopher Nolan, Blumhouse, tt0111161")
        kind = st.multiselect("Types", ["movie", "person", "company"], default=["movie", "person", "company"])
        if q:
            hit = n[n.type.isin(kind) & (n.label.astype(str).str.contains(q, case=False, regex=False) | (n.id == q))]
            hit = hit.sort_values("degree", ascending=False).head(60)
            if hit.empty:
                st.caption("No match.")
            else:
                choice = st.radio("Matches", hit.id.tolist(), format_func=lambda i: f"{g['label'][i]} · {by.at[i, 'type']}"
                                  + (f" {int(by.at[i, 'year'])}" if pd.notna(by.at[i, 'year']) else ""), index=None)
                if choice and choice != cur:
                    nav.open_node(choice)
        if len(hist) > 1:
            st.subheader("History")
            for h in reversed(hist[-12:-1]):
                if st.button(g["label"].get(h, h), key=f"hist_{h}"):
                    nav.open_node(h)

    r = by.loc[cur]
    st.subheader(f"{r.label}  ·  {r.type}" + (f"  ·  {int(r.year)}" if pd.notna(r.year) else ""))
    st.caption(f"`{cur}` · {int(r.degree):,} credits in the graph")

    if r.type == "movie":
        c = st.columns(3)
        c[0].metric("Budget", f"${r.budget_usd / 1e6:,.1f}M" if pd.notna(r.budget_usd) else "—")
        c[1].metric("Gross", f"${r.gross_usd / 1e6:,.1f}M" if pd.notna(r.gross_usd) else "—")
        c[2].metric("Gross ÷ budget", f"{r.multiple:,.2f}" if pd.notna(r.multiple) else "—")
        c = st.columns(3)
        c[0].metric("Clean sample", "yes" if r.clean else "no")
        c[1].metric("IMDb", f"{r.rating} ({int(r.votes):,} votes)" if pd.notna(r.votes) else "—")
        c[2].metric("Critics", " · ".join(x for x in [f"RT {int(r.rt)}%" if pd.notna(r.rt) else "", f"MC {int(r.mc)}" if pd.notna(r.mc) else ""] if x) or "—")
        if cur in script_index().index and st.button("Read the screenplay", type="primary"):
            nav.open_script(cur)
    links = external_links(cur)
    cols = st.columns(len(links) + 4)
    link_buttons(cur, cols[:len(links)])
    if r.type == "movie" and cur in _tagged_films(mtime(FEATURES / "tag_relevance_ids.csv") + mtime(FEATURES / "plot_tag_relevance_ids.csv")):
        if cols[len(links)].button("Tags and genres", width="stretch", help="This film's genre shares and tags."):
            nav.open_tags(cur)

    t1, t2, t3, t4 = st.tabs(["Neighbours", "Network around it", "All fields", "Raw rows"])
    with t1:
        if r.type == "movie":
            rows = e[e.target == cur].assign(id=lambda d: d.source)
        else:
            rows = e[e.source == cur].assign(id=lambda d: d.target)
        rows = rows.drop(columns=["year"], errors="ignore").merge(
            n[["id", "label", "type", "year", "multiple", "clean"]], on="id", how="left")
        cols = ["label", "type", "role", "ordering", "year", "multiple", "clean", "origin", "id"]
        rows = rows[cols].sort_values(["role", "ordering", "year"])
        st.caption("Click a row to open that node.")
        ev = st.dataframe(rows, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
                          key=f"nb_{cur}", height=min(600, 38 + 35 * len(rows)))
        if ev.selection.rows:
            nav.open_node(rows.id.iloc[ev.selection.rows[0]])
        if r.type != "movie":
            fl = rows[rows.type == "movie"].dropna(subset=["year"])
            if len(fl):
                fig = go.Figure(go.Scatter(x=fl.year, y=np.log(fl.multiple.where(fl.multiple > 0)), mode="markers",
                                           text=fl.label, customdata=fl.id, marker=dict(size=9, color="#0072B2"),
                                           hovertemplate="%{text} (%{x})<br>ln multiple %{y:.2f}<extra></extra>"))
                fig.add_hline(y=0, line_dash="dash", line_color="#999")
                fig.update_layout(clickmode="event+select", template="simple_white", height=320, title="Their films over time",
                                  yaxis_title="ln(gross / budget)")
                fig.update_layout(dragmode="select")
                ev2 = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box"), key=f"tl_{cur}")
                s = _selected_ids(ev2)
                if s:
                    nav.open_node(s[0])
            # collaborators: people/companies sharing films with this node
            films = set(fl.id) | set(rows[rows.type == "movie"].id)
            co = e[e.target.isin(films) & (e.source != cur)]
            top = (co.groupby("source").target.nunique().rename("shared_films").reset_index()
                   .merge(n[["id", "label", "type"]], left_on="source", right_on="id")
                   .sort_values("shared_films", ascending=False).head(200))
            st.markdown("**Most frequent collaborators**")
            ev3 = st.dataframe(top[["label", "type", "shared_films", "id"]], width="stretch", hide_index=True,
                               on_select="rerun", selection_mode="single-row", key=f"co_{cur}", height=320)
            if ev3.selection.rows:
                nav.open_node(top.id.iloc[ev3.selection.rows[0]])
        if r.type == "company" and (GRAPH / "companies.csv").exists():
            cc = table(GRAPH / "companies.csv")
            qid = str(r.qid) if pd.notna(r.get("qid")) else cur.split(":")[-1]
            names = cc[cc.qid.astype(str) == qid].groupby(["name", "method"]).size().rename("films").reset_index()
            st.markdown("**Infobox names merged into this company, and how each was resolved**")
            st.dataframe(names, width="stretch", hide_index=True)

    with t2:
        c1, c2 = st.columns(2)
        hops = c1.radio("Hops", [1, 2], horizontal=True, key="ego_hops")
        cap = c2.slider("Max second-hop nodes", 20, 400, 120, key="ego_cap") if hops == 2 else 0
        ids, pairs = _ego(cur, hops, cap)
        if len(ids) < 2:
            st.info("No neighbours.")
        else:
            pos = {i: k for k, i in enumerate(ids)}
            edges = np.array([[pos[a], pos[b]] for a, b in zip(pairs.source, pairs.target)])
            xy = np.array(ig.Graph(n=len(ids), edges=edges.tolist()).layout_fruchterman_reingold(niter=500).coords)
            sub = by.loc[ids]
            fig = go.Figure(_edge_trace(xy, edges[:, 0], edges[:, 1], width=0.8, opacity=.4))
            for t, colr in TYPE_COLOUR.items():
                m = (sub.type == t).to_numpy()
                if not m.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=xy[m, 0], y=xy[m, 1], mode="markers+text" if len(ids) <= 60 else "markers", name=t,
                    customdata=np.asarray(ids)[m], text=sub.label[m].astype(str).str.slice(0, 22), textposition="top center",
                    textfont=dict(size=10), hovertext=[f"{l} ({t})" for l in sub.label[m]], hoverinfo="text",
                    marker=dict(size=np.where(np.asarray(ids)[m] == cur, 22, 10), color=colr, line=dict(width=1, color="white"))))
            fig.update_layout(clickmode="event+select", dragmode="select", template="simple_white", height=640,
                              margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                              legend=dict(orientation="h"))
            st.caption(f"{len(ids):,} nodes, {len(pairs):,} links. Click a node, or drag a small box around it, to move to it.")
            ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box", "lasso"),
                                 key=f"ego_{cur}_{hops}")
            s = [x for x in _selected_ids(ev) if x != cur]
            if s:
                nav.open_node(s[0])
            others = sorted((x for x in ids if x != cur), key=lambda x: g["label"].get(x, x).lower())
            jump = st.selectbox("Or jump to", others, index=None, placeholder="Pick a node in this network…",
                                format_func=lambda x: f"{g['label'].get(x, x)} · {by.at[x, 'type']}", key=f"jump_{cur}")
            if jump:
                nav.open_node(jump)

    with t3:
        full = r.dropna().astype(str).rename("value").rename_axis("field").reset_index()
        st.dataframe(full, width="stretch", hide_index=True)
        if r.type == "movie" and (FEATURES / "film_network_features.csv").exists():
            f = table(FEATURES / "film_network_features.csv")
            fr = f[f.tconst == cur]
            if len(fr):
                st.markdown("**Network features (built from earlier films only)**")
                st.dataframe(fr.T.rename(columns={fr.index[0]: "value"}).astype(str), width="stretch")
        if r.type == "movie":
            for name in ("movies_features_10k.csv", "wiki_infobox.csv", "wiki_infobox_10k.csv", "franchise.csv",
                         "screenplay_features.csv", "review_features.csv"):
                p = FEATURES / name
                if p.exists():
                    t = table(p)
                    key = "tconst" if "tconst" in t else t.columns[0]
                    hit = t[t[key].astype(str) == cur]
                    if len(hit):
                        with st.expander(f"{name} ({len(hit)} row{'s' if len(hit) > 1 else ''})"):
                            st.dataframe(hit.T.astype(str), width="stretch")
    with t4:
        st.markdown("`nodes.csv` row")
        st.dataframe(n[n.id == cur].astype(str), width="stretch", hide_index=True)
        st.markdown("`edges.csv` rows touching this node")
        show_df(e[(e.source == cur) | (e.target == cur)].drop(columns=["key_person", "company"]), key=f"raw_{cur}")


# ---------------------------------------------------------------- screenplays

@st.cache_data(show_spinner="Indexing the MovieSum screenplays (first time only)…")
def _script_index(stamps: tuple) -> pd.DataFrame:
    rows = []
    name_re = re.compile(rb'"movie_name"\s*:\s*"((?:[^"\\]|\\.)*)"')
    id_re = re.compile(rb'"imdb_id"\s*:\s*"(tt\d+)"')
    for path in MOVIESUM:
        if not path.exists():
            continue
        with open(path, "rb") as f:
            off = 0
            for line in f:
                head = line[:600]
                m, i = name_re.search(head), id_re.search(head)
                if i:
                    rows.append({"tconst": i.group(1).decode(), "movie_name": json.loads(b'"' + m.group(1) + b'"') if m else "",
                                 "file": str(path), "offset": off, "bytes": len(line)})
                off += len(line)
    if not rows:   # no MovieSum files on this machine
        return pd.DataFrame(columns=["movie_name", "file", "offset", "bytes"], index=pd.Index([], name="tconst"))
    return pd.DataFrame(rows).drop_duplicates("tconst").set_index("tconst")


def script_index() -> pd.DataFrame:
    return _script_index(tuple(mtime(p) for p in MOVIESUM))


@st.cache_data(max_entries=8, show_spinner="Loading the screenplay…")
def _load_script(file: str, offset: int) -> dict:
    with open(file, "rb") as f:
        f.seek(offset)
        return json.loads(f.readline())


TAG = re.compile(r"<(stage_direction|scene_description|character|parenthetical|dialogue)>(.*?)</\1>", re.S)
SCENE = re.compile(r"<scene>(.*?)</scene>", re.S)
CSS = """<style>
.sp {font-family: 'Courier Prime','Courier New',monospace; font-size:14px; line-height:1.45; max-width:760px}
.sp .scn {border-top:1px solid #ddd; margin-top:14px; padding-top:6px}
.sp .no {color:#999; font-size:11px; font-family:sans-serif}
.sp .hd {font-weight:700; text-transform:uppercase; margin:6px 0}
.sp .ac {margin:6px 0}
.sp .ch {margin:10px 0 0 38%; font-weight:600}
.sp .pa {margin:0 0 0 30%; font-style:italic; color:#555}
.sp .dl {margin:0 18% 0 20%}
.sp mark {background:#ffe58a; padding:0 1px}
</style>"""
CLS = {"stage_direction": "hd", "scene_description": "ac", "character": "ch", "parenthetical": "pa", "dialogue": "dl"}


def _scenes(script: str) -> list[list[tuple[str, str]]]:
    return [[(t, html.unescape(b).strip()) for t, b in TAG.findall(block)] for block in SCENE.findall(script)]


def _render(scenes, which, query: str) -> str:
    qre = re.compile(re.escape(query), re.I) if query else None
    out = [CSS, '<div class="sp">']
    for i in which:
        out.append(f'<div class="scn"><span class="no">scene {i}</span>')
        for t, b in scenes[i]:
            if not b:
                continue
            s = html.escape(b)
            if qre:
                s = qre.sub(lambda m: f"<mark>{m.group(0)}</mark>", s)
            out.append(f'<div class="{CLS[t]}">{s}</div>')
        out.append("</div>")
    out.append("</div>")
    return "".join(out)


def _script_label(movie_name: str) -> str:
    """'Face_Off_1997' -> 'Face Off (1997)'. MovieSum joins the title words and the year with underscores."""
    title, _, year = movie_name.rpartition("_")
    return f"{title.replace('_', ' ')} ({year})" if title else movie_name


def screenplays():
    st.title("Screenplays")
    idx = script_index()
    if idx.empty:
        st.warning(f"No MovieSum files in `{DATA / 'scripts' / 'moviesum'}` (see the README: get_data.py).")
        return
    st.caption("🔒 Local only. The screenplays are copyrighted: read them here to check the parsing and the analysis, "
               "never quote or screenshot them into the paper, the site or a commit.")
    have_graph = NODES.exists()
    by = gd()["by_id"] if have_graph else None
    opts = idx.index.tolist()
    default = st.session_state.get("script_tconst")
    with st.sidebar:
        pick = st.selectbox(f"Screenplay ({len(opts):,})", opts, index=opts.index(default) if default in opts else 0,
                            format_func=lambda t: _script_label(idx.at[t, "movie_name"]), key="script_pick")
    if pick != st.session_state.get("script_tconst"):
        st.session_state["script_tconst"] = pick
        for k in ("scene_focus", "char_focus", "scene_clicked", "char_clicked"):
            st.session_state.pop(k, None)
    meta = idx.loc[pick]
    doc = _load_script(meta.file, int(meta.offset))
    scenes = _scenes(doc.get("script", ""))

    st.subheader(_script_label(meta.movie_name))
    if have_graph and pick in by.index:
        r = by.loc[pick]
        st.markdown(" · ".join(x for x in [f"**Budget** ${r.budget_usd / 1e6:,.1f}M" if pd.notna(r.budget_usd) else "**Budget** —",
                                f"**Gross** ${r.gross_usd / 1e6:,.1f}M" if pd.notna(r.gross_usd) else "**Gross** —",
                                f"**Gross ÷ budget** {r.multiple:,.2f}" if pd.notna(r.multiple) else "",
                                f"**Clean** {'yes' if r.clean else 'no'}", f"**Scenes** {len(scenes):,}"] if x).replace("$", "\\$"))
        if st.button("Open this film in the node explorer"):
            nav.open_node(pick)

    tags = Counter(t for s in scenes for t, _ in s)
    words = sum(len(b.split()) for s in scenes for t, b in s if t != "character")
    t1, t2, t4, t5 = st.tabs(["Read", "Characters", "Summary", "Markup check"])

    with t1:
        c1, c2, c3 = st.columns([2, 2, 1])
        query = c1.text_input("Search in the script", key="script_q")
        chars = Counter()
        speaker = None
        scene_chars = []
        for s in scenes:
            here = set()
            for t, b in s:
                if t == "character":
                    speaker = re.sub(r"\(.*?\)|CONT'?D", "", b, flags=re.I).strip().upper()
                elif t == "dialogue" and speaker:
                    chars[speaker] += 1
                    here.add(speaker)
            scene_chars.append(here)
        top_chars = [c for c, _ in chars.most_common(40)]
        # A character clicked in the Characters tab is handed over once; after that the box is the user's again.
        focus = st.session_state.pop("char_focus", None)
        if focus in top_chars:
            st.session_state["char_sel"] = focus
        elif st.session_state.get("char_sel") not in ["anyone"] + top_chars:
            st.session_state["char_sel"] = "anyone"
        who = c2.selectbox("Only scenes with", ["anyone"] + top_chars, key="char_sel")
        per = c3.number_input("Scenes per page", 5, 200, 25, step=5)
        cand = [i for i in range(len(scenes))
                if (who == "anyone" or who in scene_chars[i])
                and (not query or any(query.lower() in b.lower() for _, b in scenes[i]))]
        if query:
            st.caption(f"“{query}” appears in {len(cand):,} scenes.")
        if not cand:
            st.info("No scene matches.")
        else:
            start_scene = st.session_state.get("scene_focus")
            pages = max(1, (len(cand) + per - 1) // per)
            page0 = (next((k for k, i in enumerate(cand) if i >= start_scene), 0) // per) if start_scene is not None else 0
            page = st.number_input(f"Page (of {pages})", 1, pages, page0 + 1, key=f"page_{pick}_{who}_{query}_{per}_{start_scene}") - 1
            which = cand[page * per:(page + 1) * per]
            st.caption(f"Scenes {which[0]} to {which[-1]} of {len(scenes) - 1}")
            with st.container(height=760):
                st.markdown(_render(scenes, which, query), unsafe_allow_html=True)

    with t2:
        if not chars:
            st.info("No tagged dialogue in this script.")
        else:
            cdf = pd.DataFrame(chars.most_common(), columns=["character", "speeches"])
            cdf["scenes"] = cdf.character.map(lambda c: sum(c in s for s in scene_chars))
            keep = [c for c, k in chars.items() if k >= 3]
            ew = Counter()
            for s in scene_chars:
                for a, b in combinations(sorted(s & set(keep)), 2):
                    ew[(a, b)] += 1
            if len(keep) >= 2 and ew:
                pos = {c: i for i, c in enumerate(keep)}
                E = np.array([[pos[a], pos[b]] for a, b in ew])
                W = np.array(list(ew.values()), dtype=float)
                xy = np.array(ig.Graph(n=len(keep), edges=E.tolist()).layout_fruchterman_reingold(weights=W.tolist(), niter=500).coords)
                fig = go.Figure(_edge_trace(xy, E[:, 0], E[:, 1], width=1, opacity=.35))
                sz = np.array([chars[c] for c in keep], dtype=float)
                fig.add_trace(go.Scatter(x=xy[:, 0], y=xy[:, 1], mode="markers+text", text=keep, customdata=keep,
                                         textposition="top center", textfont=dict(size=10),
                                         marker=dict(size=6 + 30 * np.sqrt(sz / sz.max()), color="#E69F00",
                                                     line=dict(width=1, color="white")),
                                         hovertemplate="%{text}<extra></extra>"))
                fig.update_layout(clickmode="event+select", dragmode="select", template="simple_white", height=560,
                                  margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                                  xaxis=dict(visible=False), yaxis=dict(visible=False))
                st.caption("Characters with 3+ speeches, linked when they speak in the same scene. "
                           "Click one (or drag a box around it) to read only their scenes, or pick one in the Read tab.")
                ev = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode=("points", "box"), key=f"chars_{pick}")
                s = _selected_ids(ev)
                # The Read tab is drawn before this one, so rerun once for it to pick the click up.
                if s and s[0] != st.session_state.get("char_clicked"):
                    st.session_state["char_clicked"] = st.session_state["char_focus"] = s[0]
                    st.rerun()
                if s:
                    st.success(f"Read tab now shows only scenes with {s[0]}.")
            st.dataframe(cdf, width="stretch", hide_index=True, height=360)

    with t4:
        st.markdown(md(doc.get("summary") or "_No summary in MovieSum._"))

    with t5:
        c = st.columns(4)
        c[0].metric("Scene tags", f"{len(scenes):,}")
        c[1].metric("Speeches", f"{tags.get('dialogue', 0):,}")
        c[2].metric("Words (not cues)", f"{words:,}")
        c[3].metric("Speaker cues", f"{tags.get('character', 0):,}")
        st.dataframe(pd.DataFrame(tags.most_common(), columns=["tag", "count"]), width="stretch", hide_index=True)
        rep = OUTPUT_DIRS["scripts_text"] / "s02_parser_report.csv"
        if rep.exists():
            pr = table(rep)
            hit = pr[pr.tconst.astype(str) == pick]
            if len(hit):
                st.markdown("**Screenplay-text parser report for this script** (`s02_parser_report.csv`)")
                st.dataframe(hit.T.astype(str), width="stretch")
        n_chars = st.slider("Raw markup to show (characters)", 1000, 50000, 4000, step=1000)
        st.code(doc.get("script", "")[:n_chars], language="xml")
