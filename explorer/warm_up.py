"""Load the explorer's compiled libraries once, in the main process, before the page check or the app.

    .venv/bin/python -m explorer.warm_up

The first time a new .venv loads a compiled library (pyarrow, pandas, scikit-learn and so on) it can take
minutes on a Mac. On one fresh .venv the page check spent 161 s on Home and timed out after 600 s on Text
embeddings while importing pyarrow from its script thread; run alone afterwards, that page took 1.4 s.
Importing everything here first takes that cost in a visible step. Later runs take a few seconds.
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Compiled libraries the pages use, including the ones they import lazily (pyarrow through pd.read_parquet,
# sklearn in the embeddings map, statsmodels for plotly trend lines), then the page modules themselves.
MODULES = ["numpy", "pandas", "pyarrow", "pyarrow.parquet", "pyarrow.dataset", "scipy.stats", "sklearn.decomposition",
           "sklearn.cluster", "sklearn.manifold", "statsmodels.api", "igraph", "networkx", "plotly.graph_objects",
           "plotly.express", "streamlit", "streamlit.testing.v1",
           "explorer.pages", "explorer.graph_pages", "explorer.map_pages", "explorer.embedding_page",
           "explorer.pipeline_page", "explorer.attribution_page"]


def main() -> int:
    t0 = time.time()
    failed = []
    for name in MODULES:
        t = time.time()
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - reported, and the page check will show what it breaks
            failed.append(name)
            print(f"  can't import {name}: {type(e).__name__}: {e}", flush=True)
            continue
        if time.time() - t > 2:
            print(f"  {name}: {time.time() - t:.0f}s (first load in this .venv)", flush=True)
    print(f"Libraries loaded in {time.time() - t0:.0f}s" + (f", {len(failed)} failed" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
