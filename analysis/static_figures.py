"""Static PNG figures in the committed notebooks, so they show on GitHub (which does not run Plotly's JavaScript).

The interactive versions stay on the static pages (site/arcs, site/scripts). Two halves, used by the builders:

    use_static_figures()          # in the notebook's setup cell: figures are emitted as Plotly JSON
    pngify(nb)                    # in the builder, after execution: every Plotly JSON output becomes a PNG

The PNGs are drawn in the builder's own process rather than in the Jupyter kernel because kaleido's headless
Chrome fails to start a page from inside a kernel on macOS, while it works from a plain process.

Needs kaleido >= 1 (Plotly's image exporter) and a Chrome it can drive (`kaleido.get_chrome_sync()` fetches one).
If kaleido is missing or fails, pngify keeps the figure as an interactive Plotly output and says so.

macOS note: that Chrome may also be refused access to files under ~/Documents, where the virtualenv lives, and
kaleido loads two local scripts from there (its scopes file and plotly.js), so they are copied to the user cache
folder first and kaleido is pointed at the copies.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

import plotly.io as pio

WIDTH, SCALE = 900, 1.5
PLOTLY_MIME = "application/vnd.plotly.v1+json"


def use_static_figures() -> None:
    """Kernel side: emit figures as Plotly JSON (plus a text placeholder) for the builder to turn into PNGs."""
    pio.renderers.default = "plotly_mimetype"


def _relocate_kaleido_assets() -> None:
    """Copy kaleido's scopes script and plotly.js to the user cache folder and point kaleido at the copies."""
    import shutil  # noqa: PLC0415

    import plotly  # noqa: PLC0415
    import kaleido._page_generator as pg  # noqa: PLC0415

    if getattr(pg.PageGenerator, "_relocated", False):
        return
    cache = Path.home() / "Library" / "Caches" / "kaleido-assets"
    if not cache.parent.exists():                        # not macOS: leave kaleido as it is
        return
    cache.mkdir(parents=True, exist_ok=True)
    kjs, pjs = cache / "kaleido_scopes.js", cache / "plotly.min.js"
    shutil.copy(pg.KJS_PATH, kjs)
    shutil.copy(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", pjs)
    pg.PageGenerator.footer = pg.PageGenerator.footer.replace(pg.KJS_PATH.as_uri(), kjs.as_uri())
    original = pg.PageGenerator.__init__

    def init(self, *args, plotly=None, **kwargs):
        original(self, *args, plotly=plotly or (pjs.as_uri(), "utf-8"), **kwargs)

    pg.PageGenerator.__init__ = init
    pg.PageGenerator._relocated = True


def pngify(nb) -> int:
    """Builder side: replace every Plotly JSON output with a PNG. Returns how many figures were converted."""
    import plotly.graph_objects as go  # noqa: PLC0415

    try:
        _relocate_kaleido_assets()
    except ImportError:
        print("kaleido not installed: figures stay interactive (they will not show on GitHub)", flush=True)
        return 0
    n = 0
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if PLOTLY_MIME not in data:
                continue
            fig = go.Figure(data[PLOTLY_MIME])
            height = fig.layout.height or 450
            for attempt in range(3):   # the first page load of a cold headless Chrome sometimes fails
                try:
                    png = fig.to_image(format="png", width=fig.layout.width or WIDTH, height=height, scale=SCALE)
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 2:
                        raise
            out["data"] = {"image/png": base64.b64encode(png).decode("ascii"),
                           "text/plain": "<Plotly figure, static PNG; the interactive version is on the static page>"}
            out.setdefault("metadata", {})["image/png"] = {"width": int((fig.layout.width or WIDTH))}
            n += 1
    return n


def number_cells(nb) -> None:
    """Give code cells and their results sequential execution counts, so readers see the run order (and diffs stay
    stable, since the numbering only depends on the notebook's structure)."""
    k = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        k += 1
        cell.execution_count = k
        cell.metadata.pop("execution", None)
        for out in cell.get("outputs", []):
            if "execution_count" in out:
                out["execution_count"] = k


def pngify_file(path, attempts: int = 6) -> int:
    """Convert one written notebook in a fresh process (kaleido's Chrome is reliable on a cold start but not when
    reused across many notebooks in one long builder process). Returns the number of figures converted."""
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    for attempt in range(attempts):
        r = subprocess.run([sys.executable, "-m", "analysis.static_figures", str(path)], capture_output=True, text=True)
        if r.returncode == 0:
            return int(r.stdout.strip().splitlines()[-1])
    print(f"  could not draw PNGs for {path} after {attempts} tries; figures stay interactive", flush=True)
    return 0


PROGRESS = re.compile(r"^\s*(spaCy:|shuffles:|label permutations:|permutations:|bootstrap:)")


def drop_noise(nb) -> None:
    """Drop stderr (library warnings, which also print local install paths) and progress counters from outputs."""
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        kept = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                if out.get("name") == "stderr":
                    continue
                lines = [ln for ln in "".join(out.get("text", "")).splitlines(keepends=True) if not PROGRESS.match(ln)]
                if not "".join(lines).strip():
                    continue
                out["text"] = "".join(lines)
            kept.append(out)
        cell["outputs"] = kept


if __name__ == "__main__":
    import sys

    import nbformat as nbf

    target = sys.argv[1]
    book = nbf.read(target, as_version=4)
    count = pngify(book)
    nbf.write(book, target)
    print(count)
