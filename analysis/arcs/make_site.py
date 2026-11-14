"""Build site/arcs/index.html from the same SECTION lists and saved outputs as the notebooks.

    python -m analysis.arcs.make_site                 # the real page; refuses before Step 7 has run
    python -m analysis.arcs.make_site --draft PATH    # a preview written to PATH (never to site/), PENDING
                                                      # sections where outputs do not exist yet

Nothing is recomputed: every number comes from analysis/arcs/outputs/. Private items (script text) are left
out. Before writing, every 8-word run on the page is checked against every 8-word run of all script sentences;
if any matches, the page is NOT written.
"""
from __future__ import annotations

import datetime as dt
import html
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import mistune
import pandas as pd
import plotly.io as pio
from plotly.offline import get_plotlyjs

from analysis.scripts_text.report import Fig, Md, Private, Reading, Run, Table
from analysis.scripts_text.make_site import CSS

from .build_notebooks import STEPS
from .config import cfg, config_hash
from .paths import OUTPUTS, PRIVATE, REPO, SITE

NAV = {"01": "Sample", "02": "Text units", "03": "Instruments", "04": "Validation", "05": "Arcs",
       "06": "Order null", "07": "PRIMARY", "08": "Prediction", "09": "Robustness", "10": "Franchise",
       "11": "Reception", "12": "Back to the text", "13": "Interpretation"}
SENTIMENT_STEPS = {"03", "05", "06", "07", "08", "09", "10", "12", "13"}
md = mistune.create_markdown(plugins=["table", "strikethrough"])


def _table(df: pd.DataFrame, caption: str) -> str:
    t = df.to_html(index=False, border=0, na_rep="", escape=True, float_format=lambda v: f"{v:,.4g}")
    t = t.replace('<table class="dataframe">', f"<table><caption>{html.escape(caption)}</caption>", 1)
    return f'<div class="table-wrap">{t}</div>'


def _validated() -> bool:
    p = OUTPUTS / "s04_validation_results.json"
    return p.exists() and json.loads(p.read_text())["status"] == "validated"


def _section(num: str, module_name: str, draft: bool, validated: bool) -> str:
    mod = importlib.import_module(f"analysis.arcs.{module_name}")
    try:
        res = mod.load()
    except (FileNotFoundError, RuntimeError) as e:
        if not draft:
            raise RuntimeError(f"step {num} has no saved outputs ({e}); run it first") from e
        head = next((i.text for i in mod.SECTION if isinstance(i, Md)), f"## Step {num}")
        return md(head.split("\n\n")[0]) + f'<div class="banner"><strong>PENDING.</strong> Step {num} has not run yet.</div>'
    body, tables, figs = [], None, None
    for item in mod.SECTION:
        if isinstance(item, Md):
            body.append(md(item.text))
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
    if num in SENTIMENT_STEPS and not validated:      # after the section's opening text
        body.insert(1, '<div class="banner"><strong>NOT YET VALIDATED.</strong> The sentiment scores behind this '
                       "section have not been checked against human readers (the Step 4 sheets are empty).</div>")
    return "".join(body)


def build(draft: bool = False) -> str:
    validated = _validated()
    nav, parts = [], []
    for num, module_name, _ in STEPS:
        if importlib.util.find_spec(f"analysis.arcs.{module_name}") is None:
            continue
        nav.append(f'<li><a href="#step-{num}">{num} {html.escape(NAV[num])}</a></li>')
        parts.append(f'<section id="step-{num}">{_section(num, module_name, draft, validated)}</section>')
    top = "" if validated else ('<div class="banner"><strong>NOT YET VALIDATED.</strong> The human validation sheets '
                                "(Steps 4 and 12) are drawn but not filled. Until they are, no sentiment score on this "
                                "page has been checked by a person against the scripts.</div>")
    if draft:
        top = ('<div class="banner"><strong>DRAFT PREVIEW.</strong> Built before the primary analysis; not the '
               "published page.</div>") + top
    rq = html.escape(" ".join(cfg()["research_question"].split()))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screenplay emotional arcs and profit</title><style>{CSS}</style>
<script>{get_plotlyjs()}</script></head>
<body><header><h1>Does the emotional shape of a screenplay go with profit?</h1>
<p>{rq}</p>
<p>Part of <em>What makes a movie highly profitable?</em> (DTU 02805 Social Graphs and Interactions, autumn 2026,
Log-Log Legends). Built from the saved outputs of <code>notebooks/arcs/</code> (config {config_hash()},
{dt.date.today().isoformat()}). The primary analysis was pre-registered in <code>analysis/arcs/config.yaml</code>.
No script text appears on this page.</p>{top}</header>
<nav><ol>{"".join(nav)}</ol></nav><main>{"".join(parts)}</main>
<footer>Screenplays: MovieSum (Saxena and Keller, Findings of ACL 2024; CC BY-NC 4.0; scripts remain under copyright
and are not reproduced). Budgets and grosses: English Wikipedia infoboxes (CC BY-SA). Sentiment: VADER 3.3.2 and, where
shown, DistilBERT SST-2 and Twitter-RoBERTa run on DTU HPC. Colours: Okabe-Ito, with labels and marker shapes so no
information is carried by colour alone.</footer></body></html>"""


WORDS = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _grams(text: str, n: int = 8):
    w = WORDS.findall(text.lower())
    return (" ".join(w[i:i + n]) for i in range(len(w) - n + 1))


def check_no_text(page: str) -> int:
    """Exhaustive: every 8-word run of the page's visible text, table cells and Plotly labels is checked
    against every 8-word run of every script sentence (all 3.1M). Runs that are part of a MovieSum film name
    (an allowed identifier) are exempt. Returns the number of matching runs (must be 0)."""
    text = html.unescape(re.sub(r"<script.*?</script>", " ", page, flags=re.S))
    text = re.sub(r"<[^>]+>", " ", text)
    for js in re.findall(r'"text":\s*"([^"]*)"|"name":\s*"([^"]*)"|"title":\s*\{"text":\s*"([^"]*)"', page):
        text += " " + " ".join(js)            # figure titles, labels and hover text inside Plotly JSON
    s = pd.read_parquet(PRIVATE / "sentences.parquet", columns=["movie_name", "text"])
    titles = {g for name in s.movie_name.unique() for g in _grams(name)}   # film names are allowed identifiers
    page_grams = set(_grams(text)) - titles
    hits = set()
    for sent in s.text:
        for g in _grams(sent):
            if g in page_grams:
                hits.add(g)
    return len(hits)


def main(argv: list[str]) -> None:
    draft = "--draft" in argv
    if not draft and not (OUTPUTS / "s07_primary.json").exists():
        raise SystemExit("The page is built LAST, after the primary analysis (Step 7). Not building. "
                         "Use --draft PATH for a preview outside site/.")
    page = build(draft)
    leaks = check_no_text(page)
    if leaks:
        raise SystemExit(f"{leaks} runs of script text found in the page; NOT writing it.")
    out = Path(argv[argv.index("--draft") + 1]) if draft else SITE / "index.html"
    if draft and SITE.resolve() in out.resolve().parents:
        raise SystemExit("A draft is never written into site/.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf8")
    print(f"wrote {out} ({len(page) / 1e6:.1f} MB); script-text check: 0 runs found")


if __name__ == "__main__":
    main(sys.argv[1:])
