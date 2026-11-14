"""Build site/scripts/index.html from the same SECTION lists and saved outputs as the notebooks.

    python -m analysis.scripts_text.make_site

Nothing is recomputed: every number comes from analysis/scripts_text/outputs/. Private items (anything
showing script text) are left out, and the finished page is scanned for runs of dialogue before it is
written, so the page holds no script text.
"""
from __future__ import annotations

import datetime as dt
import html
import importlib
import re

import mistune
import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .build_notebooks import STEPS
from .config import config_hash
from .paths import REPO, SITE
from .report import Fig, Md, Private, Reading, Run, Table, takeaway_text

NAV = {"01": "Success", "02": "Text layers", "03": "Cleaning", "04": "Controls", "05": "Scattertext",
       "06": "Shuffle null", "07": "Stability", "08": "Back to the text", "09": "Second method",
       "10": "Franchise", "11": "Validation", "12": "Interpretation"}
md = mistune.create_markdown(plugins=["table", "strikethrough"])

CSS = """
:root { --ink:#1b1b1b; --muted:#5b5b5b; --line:#d9d9d9; --bg:#ffffff; --panel:#f6f6f4; --low:#D55E00; --high:#0072B2; }
* { box-sizing:border-box; }
body { margin:0; font:16px/1.6 Inter, Helvetica, Arial, sans-serif; color:var(--ink); background:var(--bg); }
header { padding:48px 24px 24px; max-width:1100px; margin:0 auto; }
header h1 { font-size:34px; line-height:1.2; margin:0 0 8px; }
header p { color:var(--muted); margin:4px 0; max-width:760px; }
nav { position:sticky; top:0; background:rgba(255,255,255,.96); border-bottom:1px solid var(--line); z-index:5; }
nav ol { display:flex; flex-wrap:wrap; gap:4px 16px; list-style:none; margin:0 auto; padding:10px 24px; max-width:1100px; font-size:14px; }
nav a { color:var(--ink); text-decoration:none; } nav a:hover { color:var(--high); text-decoration:underline; }
main { max-width:1100px; margin:0 auto; padding:0 24px 80px; }
section { border-top:1px solid var(--line); padding-top:24px; margin-top:40px; scroll-margin-top:72px; }
h2 { font-size:26px; margin:0 0 12px; } h3 { font-size:19px; }
p, li { max-width:780px; }
code { background:var(--panel); padding:1px 5px; border-radius:3px; font-size:14px; }
figure { margin:24px 0; } figcaption { color:var(--muted); font-size:14px; max-width:820px; margin-top:6px; }
.table-wrap { overflow-x:auto; margin:20px 0; } table { border-collapse:collapse; font-size:13.5px; }
th, td { border-bottom:1px solid var(--line); padding:5px 10px; text-align:left; vertical-align:top; }
th { background:var(--panel); font-weight:600; } caption { caption-side:bottom; text-align:left; color:var(--muted); font-size:14px; padding-top:6px; max-width:820px; }
.reading { background:var(--panel); border-left:4px solid var(--high); padding:12px 18px; margin:20px 0; max-width:860px; }
.takeaway { background:var(--panel); border-left:4px solid var(--accent, #009E73); padding:10px 18px; margin:14px 0 22px; max-width:860px; font-size:1.05em; }
.takeaway p { margin:0; display:inline; }
.private { border:1px dashed var(--line); color:var(--muted); padding:10px 14px; font-size:14px; max-width:860px; }
.banner { background:#fff4e5; border:1px solid #f0b46b; padding:12px 18px; margin:16px 0; max-width:860px; }
footer { color:var(--muted); font-size:13px; max-width:1100px; margin:0 auto; padding:24px; border-top:1px solid var(--line); }
@media print { nav { position:static; } section { break-inside:avoid-page; } }
"""


def _table(df: pd.DataFrame, caption: str) -> str:
    t = df.to_html(index=False, border=0, na_rep="", escape=True, float_format=lambda v: f"{v:,.4g}")
    t = t.replace('<table class="dataframe">', f"<table><caption>{html.escape(caption)}</caption>", 1)
    return f'<div class="table-wrap">{t}</div>'


def build() -> str:
    parts, nav = [], []
    validation = importlib.import_module("analysis.scripts_text.s11_validation").load()["results"]["status"]
    for num, module_name, slug in STEPS:
        mod = importlib.import_module(f"analysis.scripts_text.{module_name}")
        res = mod.load()
        tables, figs = None, None
        body = []
        title = None
        for item in mod.SECTION:
            if isinstance(item, Md):
                body.append(md(item.text))
                if title is None:
                    m = re.search(r"^## (.+)$", item.text, re.M)
                    title = m.group(1) if m else slug
            elif isinstance(item, Table):
                tables = tables or mod.tables(res)
                body.append(_table(tables[item.name], item.caption))
            elif isinstance(item, Fig):
                figs = figs or mod.figures(res)
                body.append(f"<figure>{pio.to_html(figs[item.name], full_html=False, include_plotlyjs=False)}"
                            f"<figcaption>{html.escape(item.caption)}</figcaption></figure>")
            elif isinstance(item, Reading):
                body.append(f'<div class="reading">{md(mod.reading(res))}</div>')
            elif isinstance(item, Private):
                body.append(f'<p class="private">Not shown on this page: {html.escape(item.caption)}</p>')
            elif isinstance(item, Run):
                continue
        if (take := takeaway_text(mod, res)):      # the answer first, right under the title
            body.insert(1, f'<div class="takeaway"><strong>Takeaway.</strong> {md(take)}</div>')
        nav.append(f'<li><a href="#step-{num}">{num} {html.escape(NAV[num])}</a></li>')
        parts.append(f'<section id="step-{num}">{"".join(body)}</section>')
    banner = ("" if validation != "NOT YET VALIDATED" else
              '<div class="banner"><strong>NOT YET VALIDATED.</strong> The human validation sheets (Step 11) are '
              "ready but not filled in. Until they are, nothing on this page has been checked by a person against the "
              "scripts.</div>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screenplays and profit: a Scattertext analysis</title><style>{CSS}</style>
<script>{get_plotlyjs()}</script></head>
<body><header><h1>Do profitable films use different words?</h1>
<p>The screenplay part of <em>What makes a movie highly profitable?</em> (DTU 02805 Social Graphs and Interactions,
autumn 2026, Log-Log Legends). 1,095 clean films with 100,000+ IMDb votes and a MovieSum script, compared within
genre and decade, with a shuffle null, a second method and stability checks.</p>
<p>Built from the saved outputs of <code>notebooks/scripts_text/</code> (config {config_hash()}, {dt.date.today().isoformat()}).
No script text appears on this page.</p>{banner}</header>
<nav><ol>{"".join(nav)}</ol></nav><main>{"".join(parts)}</main>
<footer>Screenplays: MovieSum (Saxena and Keller, Findings of ACL 2024; CC BY-NC 4.0; scripts remain under copyright and
are not reproduced). Budgets and grosses: English Wikipedia infoboxes (CC BY-SA). Colours: Okabe-Ito, with marker
shapes and patterns so no information is carried by colour alone.</footer></body></html>"""


def check_no_text(page: str) -> int:
    """Count eight-word runs of dialogue from 60 scripts that appear in the page (should be 0)."""
    from . import s02_layers
    lay = s02_layers.load_layers()
    hits = 0
    text = re.sub(r"<[^>]+>", " ", page)
    text = re.sub(r"\s+", " ", text)
    for d in lay.dialogue.head(60):
        words = re.findall(r"[A-Za-z']+", d)
        for i in range(0, max(0, len(words) - 8), max(1, len(words) // 5)):
            if " ".join(words[i:i + 8]) in text:
                hits += 1
    return hits


def main() -> None:
    from . import paper_figures
    paper_figures.main()
    page = build()
    leaks = check_no_text(page)
    if leaks:
        raise RuntimeError(f"{leaks} runs of script dialogue found in the page; not writing it.")
    out = SITE / "index.html"
    out.write_text(page)
    print(f"wrote {out.relative_to(REPO)} ({len(page) / 1e6:.1f} MB); dialogue check: 0 runs found")


if __name__ == "__main__":
    main()
