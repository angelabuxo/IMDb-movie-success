"""Write (and optionally execute) notebooks/arcs/NN_*.ipynb from each step's SECTION.

    python -m analysis.arcs.build_notebooks            # write and execute all steps built so far
    python -m analysis.arcs.build_notebooks 01 02      # only these
    python -m analysis.arcs.build_notebooks --no-run   # write only

Cells that show script text are tagged "private"; their outputs are cleared after execution so the
committed notebooks hold no script text.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor, TagRemovePreprocessor

from analysis.scripts_text.report import Fig, Md, Private, Reading, Run, Table

from .paths import NOTEBOOKS, REPO

# Steps are added here as they are built (step number, module, notebook slug).
STEPS = [("01", "s01_sample", "sample_outcome"), ("02", "s02_units", "text_units"),
         ("03", "s03_instruments", "instruments"), ("04", "s04_validation", "validation"),
         ("05", "s05_arcs", "arcs"), ("06", "s06_shuffle", "order_null"),
         ("07", "s07_primary", "primary"), ("08", "s08_prediction", "prediction"),
         ("09", "s09_robustness", "robustness"),
         ("10", "s10_checks", "franchise_checks"),
         ("12", "s12_back_to_text", "back_to_text"),
         ("13", "s13_interpretation", "interpretation")]

from analysis.static_figures import drop_noise, number_cells, pngify_file

SETUP = """import sys
from pathlib import Path

REPO = Path.cwd().resolve().parents[1]   # notebooks/arcs -> repo root
sys.path.insert(0, str(REPO))

import warnings

import pandas as pd
import plotly.io as pio
from IPython.display import Markdown, display

from analysis.static_figures import use_static_figures

from analysis.arcs import {module} as step

use_static_figures()   # figures become static PNGs (they show on GitHub); interactive ones are on the static page
warnings.filterwarnings("ignore", category=RuntimeWarning)   # known 0/0 on very short arcs, handled downstream
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)"""


def notebook(num: str, module_name: str) -> nbf.NotebookNode:
    mod = importlib.import_module(f"analysis.arcs.{module_name}")
    cells = [nbf.v4.new_markdown_cell(f"*Notebook {num} in `notebooks/arcs/`. Parameters (pre-registered): "
                                      "`analysis/arcs/config.yaml`; decisions and run notes: `analysis/arcs/LOG.md`. "
                                      "The static page `site/arcs/index.html` is built from the same saved outputs.*"),
             nbf.v4.new_code_cell(SETUP.format(module=module_name))]
    if not any(isinstance(i, Run) for i in mod.SECTION):
        cells.append(nbf.v4.new_code_cell("res = step.load()"))
    for item in mod.SECTION:
        if isinstance(item, Md):
            cells.append(nbf.v4.new_markdown_cell(item.text))
        elif isinstance(item, Run):
            cells.append(nbf.v4.new_code_cell(f"res = step.{getattr(item, 'call', 'run')}()   "
                                              "# recomputes this step and saves its outputs"))
        elif isinstance(item, Table):
            cells.append(nbf.v4.new_markdown_cell(f"**Table.** {item.caption}"))
            cells.append(nbf.v4.new_code_cell(f'display(step.tables(res)["{item.name}"])'))
        elif isinstance(item, Fig):
            cells.append(nbf.v4.new_markdown_cell(f"**Figure.** {item.caption}"))
            cells.append(nbf.v4.new_code_cell(f'step.figures(res)["{item.name}"].show()'))
        elif isinstance(item, Reading):
            cells.append(nbf.v4.new_markdown_cell("**Plain-language reading** (generated from the saved numbers):"))
            cells.append(nbf.v4.new_code_cell("display(Markdown(step.reading(res)))"))
        elif isinstance(item, Private):
            cells.append(nbf.v4.new_markdown_cell(f"**Private check.** {item.caption}"))
            c = nbf.v4.new_code_cell(f'display(step.private(res)["{item.name}"])   # output cleared before commit')
            c.metadata["tags"] = ["private"]
            cells.append(c)
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main(argv: list[str]) -> None:
    run = "--no-run" not in argv
    pick = [a for a in argv if not a.startswith("--")]
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    for num, module_name, slug in STEPS:
        if (pick and num not in pick) or importlib.util.find_spec(f"analysis.arcs.{module_name}") is None:
            continue
        path = NOTEBOOKS / f"{num}_{slug}.ipynb"
        nb = notebook(num, module_name)
        if run:
            print(f"executing {path.name}", flush=True)
            ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NOTEBOOKS)}})
            strip = TagRemovePreprocessor(remove_all_outputs_tags={"private"})
            strip.enabled = True
            nb, _ = strip.preprocess(nb, {})
            drop_noise(nb)
            number_cells(nb)   # sequential counts; timing metadata dropped so diffs stay quiet
        nbf.write(nb, path)
        if run:
            print(f"  {pngify_file(path)} figures drawn as PNG", flush=True)
        print(f"wrote {path.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
