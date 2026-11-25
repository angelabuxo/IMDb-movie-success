"""Shared helpers for the local explorer: paths, cached loaders, notebook-step rendering, JSON flattening.

Nothing here writes to the repo or to data/raw. Private items (script or review text) are only shown when
the viewer turns them on, and only from files that already exist on this machine.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("IMDB_DATA_DIR", REPO / "data" / "raw"))
FEATURES = DATA / "features"
OUTPUT_DIRS = {"scripts_text": REPO / "archive" / "analysis" / "scripts_text" / "outputs",
               "network": REPO / "src" / "analysis" / "network"}
TABULAR = (".csv", ".csv.gz", ".tsv", ".parquet", ".json")
log = logging.getLogger("explorer")



def missing(text: str) -> None:
    """A note that an input file isn't on this machine, with how to make it. Shown as an info box and logged as
    'missing input', so logs/explorer.log and explorer/check_pages.py list every table a page couldn't find."""
    log.warning("missing input: %s", " ".join(text.split()))
    st.info(md(text))


def md(text: str) -> str:
    """Streamlit renders $...$ as LaTeX; a $ before a digit is money here."""
    return re.sub(r"(?<!\\)\$(?=\d)", r"\\$", text)


def mtime(p: Path) -> float:
    """A file's modification time. Cached loaders take it as an argument so a rebuilt file is read again; the
    argument must not start with an underscore, or Streamlit leaves it out of the cache key."""
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner="Reading file…", max_entries=24)
def read_table(path: str, stamp: float, nrows: int | None = None) -> pd.DataFrame:
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(p)
    if name.endswith(".json"):
        return flatten_json(json.loads(p.read_text()))
    sep = "\t" if name.endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(p, sep=sep, nrows=nrows, low_memory=False)


def table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return read_table(str(path), mtime(path), nrows)


def flatten_json(obj) -> pd.DataFrame:
    """Nested results JSON as rows: every dict whose values are all scalars (or short scalar lists)
    becomes one row, keyed by the path that leads to it."""
    rows = []

    def scalar(v):
        return v is None or isinstance(v, (int, float, str, bool)) or (
            isinstance(v, list) and len(v) <= 4 and all(isinstance(x, (int, float, str)) for x in v))

    def walk(o, path):
        if isinstance(o, dict):
            if o and all(scalar(v) for v in o.values()):
                rows.append({"path": " / ".join(path), **{k: (str(v) if isinstance(v, list) else v) for k, v in o.items()}})
                return
            leaf = {k: v for k, v in o.items() if scalar(v)}
            if leaf and path:
                rows.append({"path": " / ".join(path), **{k: (str(v) if isinstance(v, list) else v) for k, v in leaf.items()}})
            elif leaf:
                rows.append({"path": "(top level)", **{k: (str(v) if isinstance(v, list) else v) for k, v in leaf.items()}})
            for k, v in o.items():
                if not scalar(v):
                    walk(v, path + [str(k)])
        elif isinstance(o, list) and o and isinstance(o[0], dict):
            for i, v in enumerate(o):
                walk(v, path + [str(i)])

    walk(obj, [])
    return pd.DataFrame(rows)


def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns that mix types (numbers and text) are shown as text, so the table always renders."""
    out = df
    for c in df.columns:
        if df[c].dtype == object and df[c].dropna().map(type).nunique() > 1:
            out = out.assign(**{c: df[c].map(lambda v: v if v is None or (isinstance(v, float) and v != v) else str(v))})
    return out


def show_df(df: pd.DataFrame, key: str, caption: str | None = None, height: int | None = None, select: bool = False):
    """A sortable table with a text filter and a CSV download. With select=True a click on a row selects it,
    and the selected row comes back as a Series (None when nothing is selected)."""
    df = arrow_safe(df)
    if caption:
        st.caption(md(caption))
    q = st.text_input("Filter rows (text match in any column)", key=f"q_{key}", placeholder="e.g. Horror, tt0111161…",
                      label_visibility="collapsed") if len(df) > 8 else ""
    view = df
    if q:
        mask = df.astype(str).apply(lambda c: c.str.contains(q, case=False, regex=False)).any(axis=1)
        view = df[mask]
        st.caption(f"{len(view):,} of {len(df):,} rows match “{q}”")
    extra = {"height": height} if height else {}
    picked = None
    if select:
        ev = st.dataframe(view, width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row",
                          key=f"sel_{key}", **extra)
        rows = ev.selection.rows if hasattr(ev, "selection") else []
        if rows and rows[0] < len(view):
            picked = view.iloc[rows[0]]
    else:
        st.dataframe(view, width="stretch", hide_index=True, **extra)
    st.download_button("Download CSV", view.to_csv(index=False).encode(), file_name=f"{key}.csv",
                       mime="text/csv", key=f"dl_{key}")
    return picked


# ---------------------------------------------------------------- notebook steps

def steps(pkg: str) -> list[tuple[str, str, str]]:
    return importlib.import_module(f"analysis.{pkg}.build_notebooks").STEPS


def unavailable(kind: str, name: str, err) -> None:
    """Why a table or figure can't be drawn here. A missing file in a private/ folder is expected on any machine
    but the one that ran the step, so it gets a quiet note naming the file instead of a warning."""
    if isinstance(err, FileNotFoundError) and "/private/" in str(err.filename or ""):
        f = Path(err.filename)
        log.info("private input not on this machine: %s %s needs %s", kind, name, f)
        shown = f.relative_to(REPO) if f.is_relative_to(REPO) else f.name
        st.caption(f"🔒 {kind} `{name}` needs `{shown}`, which holds script text or sentence-level scores and "
                   "stays on the machine that ran the step.")
    else:
        st.warning(f"{kind} `{name}` unavailable: {err if isinstance(err, Exception) else 'not produced'}")


def render_step(pkg: str, module_name: str, show_private: bool):
    """The same SECTION items the notebook and static page render, from the saved outputs."""
    from analysis.scripts_text.report import Fig, Md, Private, Reading, Run, Table, takeaway_text

    mod = importlib.import_module(f"analysis.{pkg}.{module_name}")
    try:
        res = mod.load()
    except Exception as e:  # noqa: BLE001 - shown to the user, not swallowed
        log.exception("load() failed for %s.%s", pkg, module_name)
        st.error(f"This step's saved outputs can't be loaded here: `{type(e).__name__}: {e}`.\n\n"
                 "It usually needs files in `private/` (script text, gitignored) that only exist on the machine "
                 f"that ran the step. Run it with `python -m analysis.{pkg}.{module_name}` where the data is.")
        return
    cache: dict = {}

    def get(kind):
        if kind not in cache:
            try:
                cache[kind] = getattr(mod, kind)(res)
            except Exception as e:  # noqa: BLE001
                if not (isinstance(e, FileNotFoundError) and "/private/" in str(e.filename or "")):
                    log.exception("%s() failed for %s.%s", kind, pkg, module_name)
                cache[kind] = e
        return cache[kind]

    first_md = True
    for i, item in enumerate(mod.SECTION):
        if isinstance(item, Md):
            st.markdown(md(item.text))
            if first_md:
                first_md = False
                try:
                    take = takeaway_text(mod, res)
                except Exception as e:  # noqa: BLE001
                    take = None
                    st.caption(f"Takeaway unavailable here ({type(e).__name__}: {e}); it needs the private outputs.")
                if take:
                    st.success(md("**Takeaway.** " + take))
        elif isinstance(item, Table):
            t = get("tables")
            if isinstance(t, Exception) or item.name not in t:
                unavailable("Table", item.name, t)
                continue
            show_df(t[item.name], key=f"{pkg}_{module_name}_{item.name}_{i}", caption=item.caption)
        elif isinstance(item, Fig):
            f = get("figures")
            if isinstance(f, Exception) or item.name not in f:
                unavailable("Figure", item.name, f)
                continue
            st.plotly_chart(f[item.name], width="stretch", key=f"fig_{pkg}_{module_name}_{item.name}_{i}")
            st.caption(md(item.caption))
        elif isinstance(item, Reading):
            r = get("reading")
            if isinstance(r, Exception):
                st.warning(f"Reading unavailable: {r}")
            else:
                st.info(md(r))
        elif isinstance(item, Private):
            if not show_private:
                st.caption(f"🔒 Not shown (holds script or review text): {item.caption}")
                continue
            p = get("private")
            if isinstance(p, Exception) or item.name not in p:
                st.caption(f"🔒 {item.caption}. The private files aren't on this machine "
                           f"({p if isinstance(p, Exception) else 'not produced'}).")
                continue
            st.warning("Local only: copyrighted text. Never screenshot into the paper or commit.")
            show_df(p[item.name], key=f"priv_{pkg}_{module_name}_{item.name}_{i}", caption=item.caption)
        elif isinstance(item, Run):
            continue

    prefix = module_name.split("_")[0] + "_"
    files = sorted(f for f in OUTPUT_DIRS[pkg].glob(prefix + "*") if f.name.endswith(TABULAR))
    if files:
        with st.expander(f"Raw saved outputs of this step ({len(files)} files)"):
            for f in files:
                st.markdown(f"- `{f.relative_to(REPO)}` ({f.stat().st_size / 1024:,.0f} KB)")
            st.caption("Open any of them on the **Data inspector** page to check columns, missing values and rows.")
