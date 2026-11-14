"""Write (and optionally execute) notebooks/scripts_text/NN_*.ipynb from each step's SECTION.

    python -m analysis.scripts_text.build_notebooks            # write and execute all
    python -m analysis.scripts_text.build_notebooks 05 06      # only these
    python -m analysis.scripts_text.build_notebooks --no-run   # write only

Cells that show script text are tagged "private"; their outputs are cleared after execution so the
committed notebooks hold no script text.
"""
from __future__ import annotations

import importlib
import sys

import nbformat as nbf
from nbconvert.preprocessors import ClearOutputPreprocessor, ExecutePreprocessor, TagRemovePreprocessor

from .paths import REPO
from .report import Fig, Md, Private, Reading, Run, Table

NB_DIR = REPO / "notebooks" / "scripts_text"
STEPS = [("01", "s01_groups", "success_groups"), ("02", "s02_layers", "text_layers"), ("03", "s03_clean", "cleaning"),
         ("04", "s04_controls", "controls_matching"), ("05", "s05_scatter", "scattertext"),
         ("06", "s06_null", "shuffle_null"), ("07", "s07_stability", "stability"), ("08", "s08_kwic", "back_to_text"),
         ("09", "s09_logreg", "logistic_regression"), ("10", "s10_franchise", "franchise_leakage"),
         ("11", "s11_validation", "validation"), ("12", "s12_interpretation", "interpretation")]

from analysis.static_figures import drop_noise, number_cells, pngify_file

SETUP = """import sys
from pathlib import Path

REPO = Path.cwd().resolve().parents[1]   # notebooks/scripts_text -> repo root
sys.path.insert(0, str(REPO))

import warnings

import pandas as pd
import plotly.io as pio
from IPython.display import Markdown, display

from analysis.static_figures import use_static_figures

from analysis.scripts_text import {module} as step

use_static_figures()   # figures become static PNGs (they show on GitHub); interactive ones are on the static page
warnings.filterwarnings("ignore", category=RuntimeWarning)   # known 0/0 on very short arcs, handled downstream
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)"""


def notebook(num: str, module_name: str) -> nbf.NotebookNode:
    mod = importlib.import_module(f"analysis.scripts_text.{module_name}")
    cells = [nbf.v4.new_markdown_cell(f"*Notebook {num} of 12 in `notebooks/scripts_text/`. Parameters: "
                                      "`analysis/scripts_text/config.yaml`; decisions and run notes: "
                                      "`analysis/scripts_text/LOG.md`. The static page `site/scripts/index.html` "
                                      "is built from the same outputs.*"),
             nbf.v4.new_code_cell(SETUP.format(module=module_name))]
    has_run = any(isinstance(i, Run) for i in mod.SECTION)
    if not has_run:
        cells.append(nbf.v4.new_code_cell("res = step.load()"))
    for item in mod.SECTION:
        if isinstance(item, Md):
            cells.append(nbf.v4.new_markdown_cell(item.text))
        elif isinstance(item, Run):
            cells.append(nbf.v4.new_code_cell("res = step.run()   # recomputes this step and saves its outputs"))
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
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for num, module_name, slug in STEPS:
        if pick and num not in pick:
            continue
        path = NB_DIR / f"{num}_{slug}.ipynb"
        nb = notebook(num, module_name)
        if run:
            print(f"executing {path.name}", flush=True)
            ExecutePreprocessor(timeout=7200, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB_DIR)}})
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
