"""Step 2: extract dialogue, action and full-text layers from the MovieSum markup."""
from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import s01_groups
from .config import cfg, log_run
from .paths import MOVIESUM, OUTPUTS, PRIVATE
from .report import HIGH, LOW, NEUTRAL, Fig, Md, Private, Reading, Run, Table, style

LAYERS = PRIVATE / "layers.parquet"          # script text: never committed
REPORT = OUTPUTS / "s02_parser_report.csv"   # counts only
SUMMARY = OUTPUTS / "s02_summary.json"

SCENE = re.compile(r"<scene>(.*?)</scene>", re.S)
TAG = re.compile(r"<(stage_direction|scene_description|character|parenthetical|dialogue)>(.*?)</\1>", re.S)
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
EXAMPLES = ["tt0110912", "tt1375666", "tt0068646"]   # Pulp Fiction, Inception, The Godfather


def parse(script: str) -> dict:
    """Split one script into layers. Headings (stage_direction) go to the full text only; character
    cues go nowhere (they are names, handled in Step 3), but are kept as a list for the name step."""
    dialogue, action, full, headings, cues = [], [], [], [], []
    scenes = SCENE.findall(script)
    for block in scenes:
        for tag, body in TAG.findall(block):
            body = html.unescape(body).strip()
            if not body:
                continue
            if tag == "character":
                cues.append(body)
                continue
            full.append(body)
            if tag == "dialogue":
                dialogue.append(body)
            elif tag == "scene_description":
                action.append(body)
            elif tag == "stage_direction":
                headings.append(body)
    return {"dialogue": "\n".join(dialogue), "action": "\n".join(action), "full": "\n".join(full),
            "headings": "\n".join(headings), "cues": cues, "n_scenes": len(scenes),
            "n_dialogue_lines": len(dialogue), "n_action_blocks": len(action), "n_headings": len(headings)}


def run() -> dict:
    sc = s01_groups.load()["scripts"]
    want = set(sc.tconst)
    best = {}
    for path in MOVIESUM:
        with open(path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if r["imdb_id"] in want and len(r["script"] or "") > len((best.get(r["imdb_id"]) or {}).get("script", "")):
                    best[r["imdb_id"]] = r
    rows, report = [], []
    for tid in sorted(want):
        raw = best[tid]["script"] or ""
        try:
            p = parse(raw)
            err = None
        except Exception as e:  # counted, never silently dropped
            p, err = {"dialogue": "", "action": "", "full": "", "headings": "", "cues": [], "n_scenes": 0,
                      "n_dialogue_lines": 0, "n_action_blocks": 0, "n_headings": 0}, repr(e)
        rows.append({"tconst": tid, **{k: p[k] for k in ("dialogue", "action", "full", "headings")},
                     "cues": json.dumps(p["cues"])})
        rec = {"tconst": tid, "parse_error": err, "raw_chars": len(raw), "n_scenes": p["n_scenes"],
               "n_dialogue_lines": p["n_dialogue_lines"], "n_action_blocks": p["n_action_blocks"],
               "n_headings": p["n_headings"], "n_cues": len(p["cues"])}
        for layer in ("dialogue", "action", "full"):
            rec[f"{layer}_chars"] = len(p[layer])
            rec[f"{layer}_words"] = len(WORD.findall(p[layer]))
        report.append(rec)
    pd.DataFrame(rows).to_parquet(LAYERS, index=False)
    rep = pd.DataFrame(report).merge(sc[["tconst", "group_A", "group_B", "group_C"]], on="tconst")
    rep.to_csv(REPORT, index=False)
    flag = cfg()["text"]["min_scenes_flag"]
    summary = {
        "scripts": int(len(rep)), "parse_errors": int(rep.parse_error.notna().sum()),
        "empty_dialogue": int((rep.dialogue_words == 0).sum()), "empty_action": int((rep.action_words == 0).sum()),
        "under_min_scenes": int((rep.n_scenes < flag).sum()), "min_scenes_flag": flag,
        "excluded_little_dialogue": int((rep.dialogue_words < cfg()["text"]["min_dialogue_words"]).sum()),
        "excluded_little_dialogue_titles": sorted(rep.merge(sc[["tconst", "movie_name"]], on="tconst")
                                                  .query(f"dialogue_words < {cfg()['text']['min_dialogue_words']}").movie_name),
        "dialogue_share_of_words_median": float((rep.dialogue_words / rep.full_words).median()),
        "median_words": {l: int(rep[f"{l}_words"].median()) for l in ("dialogue", "action", "full")},
        "total_words": {l: int(rep[f"{l}_words"].sum()) for l in ("dialogue", "action", "full")},
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s02_layers", [REPORT.name, SUMMARY.name], {"layers": cfg()["text"]["layers"], "min_scenes_flag": flag})
    return load()


def load() -> dict:
    return {"report": pd.read_csv(REPORT, dtype={"tconst": str}), "summary": json.loads(SUMMARY.read_text())}


def load_layers() -> pd.DataFrame:
    return pd.read_parquet(LAYERS)


# ---------------------------------------------------------------- tables

def _sanity(rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mask in (("parse error", rep.parse_error.notna()), ("empty dialogue layer", rep.dialogue_words == 0),
                       ("empty action layer", rep.action_words == 0),
                       (f"fewer than {cfg()['text']['min_scenes_flag']} scene tags (flagged, kept)",
                        rep.n_scenes < cfg()["text"]["min_scenes_flag"]),
                       (f"fewer than {cfg()['text']['min_dialogue_words']} dialogue words (excluded from all "
                        "comparisons: dialogue markup failed)", rep.dialogue_words < cfg()["text"]["min_dialogue_words"]),
                       ("no character cues", rep.n_cues == 0)):
        rows.append({"check": name, "scripts": int(mask.sum()), "share": round(float(mask.mean()), 3)})
    return pd.DataFrame(rows)


def _lengths(rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k in "ABC":
        for grp in ("low", "high"):
            g = rep[rep[f"group_{k}"] == grp]
            rows.append({"definition": k, "group": grp, "scripts": len(g),
                         **{f"median {l} words": int(g[f"{l}_words"].median()) for l in ("dialogue", "action", "full")},
                         "median scenes": int(g.n_scenes.median())})
    return pd.DataFrame(rows)


def _examples(rep: pd.DataFrame) -> pd.DataFrame:
    """Before/after as counts: raw markup size against what each layer keeps. No text."""
    e = rep[rep.tconst.isin(EXAMPLES)].copy()
    names = s01_groups.load()["scripts"].set_index("tconst").movie_name
    e.insert(0, "title", e.tconst.map(names))
    return e[["title", "raw_chars", "n_scenes", "n_headings", "n_cues", "n_dialogue_lines", "n_action_blocks",
              "dialogue_words", "action_words", "full_words"]]


def tables(res: dict) -> dict:
    rep = res["report"]
    return {"sanity": _sanity(rep), "lengths": _lengths(rep), "examples": _examples(rep)}


def private(res: dict) -> dict:
    """The first 12 words of each layer for the three examples. Script text: notebook only."""
    lay = load_layers().set_index("tconst")
    names = s01_groups.load()["scripts"].set_index("tconst").movie_name
    rows = []
    for tid in EXAMPLES:
        for layer in ("headings", "dialogue", "action"):
            words = lay.loc[tid, layer].split()[:12]
            rows.append({"title": names[tid], "layer": layer, "first 12 words": " ".join(words)})
    return {"examples_text": pd.DataFrame(rows)}


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    rep = res["report"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["dialogue", "action", "full text"], shared_yaxes=True)
    for i, layer in enumerate(("dialogue", "action", "full"), 1):
        for grp, colour, name in (("low", LOW, "unsuccessful (C)"), ("high", HIGH, "successful (C)"),
                                  (None, NEUTRAL, "not in (C)")):
            v = rep[(rep.group_C == grp) if grp else ~rep.group_C.isin(["low", "high"])][f"{layer}_words"]
            fig.add_trace(go.Box(y=v, name=f"{name}", marker_color=colour, showlegend=i == 1,
                                 boxpoints="outliers"), row=1, col=i)
        fig.update_xaxes(showticklabels=False, row=1, col=i)
    fig.update_yaxes(title_text="words per script", row=1, col=1)
    style(fig, "Script length by layer and group", height=400)

    sh = go.Figure(go.Histogram(x=rep.dialogue_words / rep.full_words, nbinsx=50, marker_color=NEUTRAL))
    sh.update_layout(xaxis_title="dialogue words ÷ all words in the script", yaxis_title="scripts")
    style(sh, "How much of each script is dialogue", height=340)
    return {"lengths": fig, "dialogue_share": sh}


def reading(res: dict) -> str:
    s = res["summary"]
    return (f"All {s['scripts']:,} scripts parse ({s['parse_errors']} errors). {s['empty_dialogue']} have an empty "
            f"dialogue layer and {s['empty_action']} an empty action layer. {s['under_min_scenes']} have fewer than "
            f"{s['min_scenes_flag']} scene tags, which usually means the scene markup is missing; they stay in, "
            f"flagged, because the dialogue is still tagged. {s['excluded_little_dialogue']} scripts have fewer than "
            f"{cfg()['text']['min_dialogue_words']} tagged dialogue words, which means the dialogue markup failed "
            f"({', '.join(s['excluded_little_dialogue_titles'])}); they are excluded from every comparison from "
            f"Step 5 on and counted here. The median script has "
            f"{s['median_words']['dialogue']:,} dialogue words, {s['median_words']['action']:,} action words and "
            f"{s['median_words']['full']:,} words in all; dialogue is a median {s['dialogue_share_of_words_median']:.0%} "
            f"of the words. Across all scripts that is {s['total_words']['dialogue']:,} dialogue words and "
            f"{s['total_words']['action']:,} action words.")


SECTION = [
    Md("""## Step 2 · Extracting text from the MovieSum markup

**What we do.** MovieSum marks up each screenplay: a `<scene>` holds a heading (`<stage_direction>`, e.g. INT.
DINER - DAY), action lines (`<scene_description>`), speaker cues (`<character>`), actors' directions
(`<parenthetical>`) and lines of speech (`<dialogue>`). We split every script into three layers:

1. **dialogue**: only what characters say;
2. **action**: only the descriptive prose of what we see;
3. **full**: everything except speaker cues, in script order.

**Why.** They answer different questions. Dialogue is how characters speak; action is how the writer tells the
story on the page. A word can be distinctive in one layer and invisible in the other. The primary analysis uses
dialogue because it is closest to what an audience hears.

**What it measures.** Which words sit in which part of the script, as MovieSum's tagger labelled them.

**What it misses.** The tagger isn't perfect: a cue it missed lands in the dialogue, a page header lands in the
action. Extraction is not resolution: a cue like "JULES (V.O.)" is just a string here. Speaker cues are set
aside, not lost: Step 3 uses them to find character names."""),
    Run("s02_layers"),
    Table("s02_layers", "sanity", "Parser sanity report on the 1,095 scripts. Nothing is dropped here; every "
          "problem is counted."),
    Table("s02_layers", "examples", "Before and after, as counts, for three scripts: the raw markup's size, "
          "and what each layer keeps."),
    Private("s02_layers", "examples_text", "The first 12 words of each layer for the same three scripts, to check "
            "that headings, dialogue and action land in the right layer. Notebook only, output cleared before "
            "commit (copyright)."),
    Fig("s02_layers", "lengths", "Words per script in each layer, for the unsuccessful (orange) and successful "
        "(blue) groups of definition (C) and the films in neither. If one group's scripts were much longer, "
        "raw counts would mislead; this is why every comparison uses rates per 10,000 words."),
    Fig("s02_layers", "dialogue_share", "Dialogue's share of each script's words. The spread shows why the "
        "layers can disagree: some scripts are mostly talk, some mostly description."),
    Table("s02_layers", "lengths", "Median words and scenes per script, by group and definition."),
    Reading("s02_layers"),
]
