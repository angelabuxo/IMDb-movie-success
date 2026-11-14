"""One description of each section, rendered twice: as a notebook (build_notebooks.py) and as the
static page (make_site.py). Both read the same saved outputs, so their numbers cannot diverge.

A step module defines SECTION, a list of items:

    Md(text)                         markdown, identical in notebook and page
    Run("s01_groups")                notebook only: recompute the step and save its outputs
    Table("s01_groups", "sizes", caption)    a DataFrame from module.tables(res)[name]
    Fig("s01_groups", "budget", caption)     a Plotly figure from module.figures(res)[name]
    Reading("s01_groups")            markdown built from the saved numbers: module.reading(res)
    Private(...)                     like Table, but holds script text: shown in the notebook, its
                                     output stripped before commit, never put on the page

and load() -> res, which reads the step's saved outputs.
"""
from __future__ import annotations

from dataclasses import dataclass

# Colour-blind-safe (Okabe-Ito). Groups are also told apart by marker shape and label, never by
# colour alone.
LOW, HIGH, NEUTRAL, ACCENT = "#D55E00", "#0072B2", "#999999", "#009E73"
LOW_SYMBOL, HIGH_SYMBOL = "triangle-down", "circle"
FONT = dict(family="Inter, Helvetica, Arial, sans-serif", size=13)


@dataclass
class Md:
    text: str


@dataclass
class Run:
    module: str


@dataclass
class Table:
    module: str
    name: str
    caption: str


@dataclass
class Fig:
    module: str
    name: str
    caption: str


@dataclass
class Reading:
    module: str


@dataclass
class Private:
    module: str
    name: str
    caption: str


def style(fig, title=None, height=420):
    """House style for every Plotly figure."""
    fig.update_layout(template="simple_white", font=FONT, height=height + 20, margin=dict(l=70, r=30, t=90, b=60),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                      title=dict(text=title, x=0.01, xanchor="left", y=0.985, yanchor="top", yref="container"))
    return fig


def takeaway_text(mod, res=None) -> str | None:
    """The step's one-to-three-sentence takeaway, shown right under its title in the notebook and on the page.
    A step opts in by defining takeaway(res); None when it has none or its outputs aren't saved yet."""
    if not hasattr(mod, "takeaway"):
        return None
    try:
        return mod.takeaway(res if res is not None else mod.load())
    except (FileNotFoundError, RuntimeError, KeyError):
        return None


def nb_md(text: str) -> str:
    """Markdown for a notebook cell: a "$" before a digit is a dollar amount, not the start of a math formula,
    so escape it (Jupyter and GitHub otherwise render "$40M ... $25M" as math)."""
    import re
    return re.sub(r"(?<!\\)\$(?=\d)", r"\\$", text)
