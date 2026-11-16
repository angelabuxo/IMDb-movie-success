"""Logging for the local explorer: every crash, error, warning and missing-file note goes to logs/explorer.log.

app.py calls setup() once per run and wraps each page with logged(). After that:

- a page that raises is logged with its traceback, and the traceback is still shown in the app;
- every st.error and st.warning a page shows is also written to the log, with the page it came from;
- Python warnings (pandas, numpy, plotly deprecations) go to the log instead of only the terminal;
- each page view is logged with how long it took, so slow pages show up.

The log rotates at 2 MB and keeps 3 old files. logs/ is gitignored. Set EXPLORER_LOG_LEVEL=DEBUG for more.
"""
from __future__ import annotations

import functools
import logging
import logging.handlers
import os
import time
from pathlib import Path

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = Path(os.environ.get("EXPLORER_LOG_DIR", REPO / "logs"))
LOG_FILE = LOG_DIR / "explorer.log"
log = logging.getLogger("explorer")
# The page being drawn lives on the logger, a process-wide singleton, not in this module: Streamlit reloads
# modules when files change, and a filter from the old module must still see the new page name.
if not hasattr(log, "explorer_page"):
    log.explorer_page = "-"


def setup() -> logging.Logger:
    """Configure the 'explorer' logger once per process (Streamlit reruns the script on every click)."""
    if getattr(log, "_explorer_ready", False):
        return log
    log.setLevel(os.environ.get("EXPLORER_LOG_LEVEL", "INFO").upper())
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [%(page)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    class PageFilter(logging.Filter):
        def filter(self, record):
            record.page = getattr(record, "page", log.explorer_page)
            return True

    log.addFilter(PageFilter())          # tags every record with the page it came from, for any handler
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.addFilter(PageFilter())
        log.addHandler(fh)
    except OSError as e:  # a read-only checkout still runs, logging to the terminal only
        print(f"explorer: can't write {LOG_FILE} ({e}); logging to the terminal only")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.addFilter(PageFilter())
    sh.setLevel(logging.WARNING)
    log.addHandler(sh)

    logging.captureWarnings(True)
    pyw = logging.getLogger("py.warnings")
    for h in log.handlers:
        pyw.addHandler(h)
    pyw.propagate = False

    _patch_streamlit()
    log._explorer_ready = True
    log.info("explorer started, data folder %s", os.environ.get("IMDB_DATA_DIR", REPO / "data" / "raw"))
    return log


def _patch_streamlit():
    """Mirror st.error and st.warning (and the same on any container) into the log, so a message seen on screen is also in the file."""
    if getattr(st, "_explorer_logged", False):
        return
    for name, level in (("error", logging.ERROR), ("warning", logging.WARNING)):
        orig = getattr(st, name)

        @functools.wraps(orig)
        def wrapper(body, *a, _orig=orig, _level=level, **kw):
            log.log(_level, "shown: %s", " ".join(str(body).split())[:1000])
            return _orig(body, *a, **kw)

        setattr(st, name, wrapper)
        # the same on containers (st.sidebar.warning, a tab's or column's .error); st.* above is a bound method
        # captured at import, so it needs its own wrapper
        cls_orig = getattr(DeltaGenerator, name)

        @functools.wraps(cls_orig)
        def cls_wrapper(self, body, *a, _orig=cls_orig, _level=level, **kw):
            log.log(_level, "shown: %s", " ".join(str(body).split())[:1000])
            return _orig(self, body, *a, **kw)

        setattr(DeltaGenerator, name, cls_wrapper)
    st._explorer_logged = True


def logged(fn, name: str | None = None):
    """Wrap a page function: log the view and its time, and log (then re-show) any exception."""
    label = name or fn.__name__

    @functools.wraps(fn)
    def page():
        log.explorer_page = label
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - logged, then shown as Streamlit would. Reruns and stops are
            # BaseException in Streamlit, so st.switch_page and st.rerun pass through untouched.
            log.exception("page crashed: %s: %s", type(e).__name__, e)
            st.exception(e)
        else:
            log.debug("drawn in %.1fs", time.time() - t0)
            if time.time() - t0 > 10:
                log.info("slow page: %.1fs", time.time() - t0)

    return page
