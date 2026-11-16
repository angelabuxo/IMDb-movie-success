"""explorer/log.py: a crashing page, st.error, st.warning and a missing input file all reach logs/explorer.log."""
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

SCRIPT = """
from explorer.log import logged, setup
from explorer.common import missing
import streamlit as st
setup()

def page():
    missing("Needs `features/nothing.csv`.")
    st.warning("a caveat")
    st.error("an error")
    st.sidebar.warning("a sidebar warning")
    if st.session_state.get("boom"):
        raise ValueError("planted crash")

logged(page, "demo")()
"""


class Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append((record.levelname, getattr(record, "page", "-"), record.getMessage()))


@pytest.fixture
def collect():
    h = Collect()
    logging.getLogger("explorer").addHandler(h)
    yield h.lines
    logging.getLogger("explorer").removeHandler(h)


def test_messages_shown_are_logged_with_their_page(collect):
    at = AppTest.from_string(SCRIPT, default_timeout=60).run()
    assert not at.exception
    assert len(at.info) == 1 and len(at.warning) == 2 and len(at.error) == 1   # still shown on the page
    got = {(lvl, msg) for lvl, page, msg in collect if page == "demo"}
    assert ("WARNING", "missing input: Needs `features/nothing.csv`.") in got
    assert ("WARNING", "shown: a caveat") in got
    assert ("ERROR", "shown: an error") in got
    assert ("WARNING", "shown: a sidebar warning") in got


def test_a_crash_is_logged_with_traceback_and_still_shown(collect):
    at = AppTest.from_string(SCRIPT, default_timeout=60)
    at.session_state["boom"] = True
    at.run()
    assert at.exception and "planted crash" in at.exception[0].value
    assert any(lvl == "ERROR" and msg.startswith("page crashed: ValueError: planted crash") for lvl, _, msg in collect)


def test_setup_is_idempotent():
    from explorer.log import log, setup
    setup()
    n = len(log.handlers)
    setup()
    assert len(log.handlers) == n
