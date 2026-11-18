"""Local explorer for the project. Run from the repo root:

    streamlit run explorer/app.py

It reads data/raw (or $IMDB_DATA_DIR) and the committed outputs, and writes nothing. Not meant to be deployed:
the data folder holds licensed and copyrighted material.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from explorer import attribution_page, embedding_page, graph_pages, map_pages, nav, pages, pipeline_page  # noqa: E402
from explorer.log import logged, setup  # noqa: E402

setup()

st.set_page_config(page_title="Movie profit explorer", page_icon="🎬", layout="wide")

P = {
    "home": st.Page(logged(pages.home), title="Home", icon=":material/home:", url_path="home", default=True),
    "pipeline": st.Page(logged(pipeline_page.pipeline), title="Data pipeline", icon=":material/account_tree:",
                        url_path="pipeline"),
    "text": st.Page(logged(pages.text), title="Screenplay text", icon=":material/text_fields:", url_path="text"),
    "embeddings": st.Page(logged(embedding_page.embeddings), title="Text embeddings", icon=":material/scatter_plot:",
                          url_path="embeddings"),
    "network": st.Page(logged(pages.network), title="Network results", icon=":material/hub:", url_path="network"),
    "graph": st.Page(logged(graph_pages.whole_graph), title="Whole graph", icon=":material/scatter_plot:", url_path="graph"),
    "movie_network": st.Page(logged(graph_pages.movie_network), title="Film-only network", icon=":material/hub:",
                             url_path="movie-network"),
    "node": st.Page(logged(graph_pages.node), title="Node explorer", icon=":material/account_tree:", url_path="node"),
    "scripts": st.Page(logged(graph_pages.screenplays), title="Screenplays", icon=":material/description:", url_path="scripts"),
    "maps": st.Page(logged(map_pages.maps), title="Embedding maps", icon=":material/bubble_chart:", url_path="maps"),
    "mapchecks": st.Page(logged(map_pages.checks), title="Embedding map checks", icon=":material/rule:",
                         url_path="map-checks"),
    "profit": st.Page(logged(pages.profit), title="Profit playground", icon=":material/insights:", url_path="profit"),
    "attribution": st.Page(logged(attribution_page.attribution), title="Profit attribution", icon=":material/pie_chart:",
                           url_path="attribution"),
    "calendar": st.Page(logged(pages.calendar), title="Release calendar", icon=":material/calendar_month:", url_path="calendar"),
    "tags": st.Page(logged(pages.tags), title="Tags and genres", icon=":material/sell:", url_path="tags"),
    "followups": st.Page(logged(pages.tag_followups), title="Tags: follow-ups", icon=":material/experiment:", url_path="tag-followups"),
    "data": st.Page(logged(pages.inspector), title="Data inspector", icon=":material/table_view:", url_path="data"),
    "checks": st.Page(logged(pages.checks), title="Checks", icon=":material/fact_check:", url_path="checks"),
    "tmdb": st.Page(logged(pages.tmdb), title="Second money source", icon=":material/compare_arrows:", url_path="tmdb"),
}
nav.PAGES.clear()
nav.PAGES.update(P)

st.navigation({
    "Overview": [P["home"], P["pipeline"]],
    "Notebooks": [P["text"], P["embeddings"], P["maps"], P["mapchecks"], P["network"]],
    "Graph and scripts": [P["graph"], P["movie_network"], P["node"], P["scripts"]],
    "Explore and check": [P["profit"], P["attribution"], P["calendar"], P["tags"], P["followups"], P["data"],
                          P["checks"], P["tmdb"]],
}).run()
