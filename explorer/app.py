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

from explorer import attribution_page, graph_pages, nav, pages  # noqa: E402

st.set_page_config(page_title="Movie profit explorer", page_icon="🎬", layout="wide")

P = {
    "home": st.Page(pages.home, title="Home", icon=":material/home:", url_path="home", default=True),
    "arcs": st.Page(pages.arcs, title="Screenplay arcs", icon=":material/show_chart:", url_path="arcs"),
    "text": st.Page(pages.text, title="Screenplay text", icon=":material/text_fields:", url_path="text"),
    "network": st.Page(pages.network, title="Network results", icon=":material/hub:", url_path="network"),
    "graph": st.Page(graph_pages.whole_graph, title="Whole graph", icon=":material/scatter_plot:", url_path="graph"),
    "node": st.Page(graph_pages.node, title="Node explorer", icon=":material/account_tree:", url_path="node"),
    "scripts": st.Page(graph_pages.screenplays, title="Screenplays", icon=":material/description:", url_path="scripts"),
    "profit": st.Page(pages.profit, title="Profit playground", icon=":material/insights:", url_path="profit"),
    "attribution": st.Page(attribution_page.attribution, title="Profit attribution", icon=":material/pie_chart:",
                           url_path="attribution"),
    "data": st.Page(pages.inspector, title="Data inspector", icon=":material/table_view:", url_path="data"),
    "checks": st.Page(pages.checks, title="Checks", icon=":material/fact_check:", url_path="checks"),
    "tmdb": st.Page(pages.tmdb, title="Second money source", icon=":material/compare_arrows:", url_path="tmdb"),
}
nav.PAGES.clear()
nav.PAGES.update(P)

st.navigation({
    "Overview": [P["home"]],
    "Notebooks": [P["arcs"], P["text"], P["network"]],
    "Graph and scripts": [P["graph"], P["node"], P["scripts"]],
    "Explore and check": [P["profit"], P["attribution"], P["data"], P["checks"], P["tmdb"]],
}).run()
