"""Cross-page links. app.py registers its st.Page objects here on every run; pages call go()."""
from __future__ import annotations

import streamlit as st

PAGES: dict = {}


def go(page: str, **state):
    """Store state for the target page and switch to it (or just rerun when pages aren't registered)."""
    st.session_state.update(state)
    if page in PAGES:
        st.switch_page(PAGES[page])
    else:
        st.rerun()


def open_node(node_id: str):
    hist = st.session_state.setdefault("node_history", [])
    if not hist or hist[-1] != node_id:
        hist.append(node_id)
    go("node", node_id=node_id)


def open_script(tconst: str):
    go("scripts", script_tconst=tconst)
