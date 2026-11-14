"""Step 12: back to the text. For the scripts with the most extreme primary descriptors, find the scenes at
the arc's minimum and maximum and the sentence that drives each, and lay them out for a human to check.
No outcome is used. If code output and text disagree, the text wins.

Selection (fixed 2026-11-14, LOG.md): D1 the 10 most volatile primary scripts; D2 the 5 most negative and 5
most positive. Arc minimum / maximum = argmin / argmax of the smoothed arc; the driving scene is the most
extreme scene (by scene score) among the scenes under the smoothing window there; the driving sentence is the
most extreme sentence in that scene. Window = that sentence if <= 15 words, else the 15 words centred on its
strongest VADER lexicon word.

Committed: outputs/s12_back_to_text.csv (keys and numbers, empty human columns, no text) and
outputs/s12_driver_words.csv (single lexicon words counted over the driving sentences).
Private: private/back_to_text/windows.csv and .html (the <= 15-word windows)."""
from __future__ import annotations

import html
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from analysis.scripts_text.report import Md, Private, Reading, Run, Table

from . import s02_units, s03_instruments, s05_arcs
from .config import cfg, log_run
from .paths import OUTPUTS, PRIVATE

SHEET = OUTPUTS / "s12_back_to_text.csv"
WORDS = OUTPUTS / "s12_driver_words.csv"
META = OUTPUTS / "s12_meta.json"
PRIV = PRIVATE / "back_to_text"
HUMAN = ["score_matches_text", "driver", "notes"]
DRIVERS = ["described event (death, killing, ...)", "tokenisation/markup artefact", "tone", "other"]
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _instrument() -> tuple[str, str]:
    d = OUTPUTS / "s03_instrument_decision.json"
    if d.exists():
        return json.loads(d.read_text())["primary_instrument"], "primary instrument (fallback rule)"
    return "vader", "VADER: the fallback rule has not been applied yet; rerun if the contextual model becomes primary"


def _select(desc: pd.DataFrame) -> pd.DataFrame:
    n = cfg()["back_to_text"]["extreme_scripts_per_descriptor"]
    p = desc[desc.in_primary]
    a = p.nlargest(n, "D1").assign(descriptor="D1", side="most volatile")
    b = p.nsmallest(n // 2, "D2").assign(descriptor="D2", side="most negative (ends darker)")
    c = p.nlargest(n - n // 2, "D2").assign(descriptor="D2", side="most positive (ends lighter)")
    out = pd.concat([a, b, c])
    out["rank"] = out.groupby(["descriptor", "side"]).cumcount() + 1
    return out


def _window(text: str, lex: dict, n: int) -> str:
    w = text.split()
    if len(w) <= n:
        return text
    val = [abs(lex.get(TOKEN.sub(lambda m: m.group(0).lower(), x).strip(".,!?;:\"'()").lower(), 0.0)) for x in w]
    k = int(np.argmax(val))
    lo = max(0, min(k - n // 2, len(w) - n))
    return " ".join(w[lo:lo + n])


def run() -> dict:
    c = cfg()
    inst, why = _instrument()
    a = c["arc"]
    T, h = a["time_points"], a["smoothing_window"] // 2
    if SHEET.exists():
        old = pd.read_csv(SHEET, dtype=str, keep_default_na=False)
        if (old[HUMAN] != "").any().any():
            raise RuntimeError("s12_back_to_text.csv has human entries; refusing to overwrite it.")
    desc = s05_arcs.load(inst)["desc"]
    arcs = s05_arcs.load(inst)["arcs"]
    scenes = s03_instruments.load_scores(inst, "scene").sort_values(["movie_name", "scene"])
    sents = s02_units.load_sentences()
    sc_scores = s03_instruments.load_scores(inst, "sentence")
    sents = sents.assign(score=sc_scores[inst].to_numpy())
    others = {i: s03_instruments.load_scores(i, "scene") for i in ("vader", "sst2", "roberta3", "modernbert3")
              if i != inst and (OUTPUTS / f"s03_scene_{i}.csv.gz").exists()}
    lex = s03_instruments._vader().lexicon
    rows, priv, words = [], [], {"arc minimum": Counter(), "arc maximum": Counter()}
    for r in _select(desc).itertuples():
        g = scenes[scenes.movie_name == r.movie_name]
        x = g[f"{inst}_both"].to_numpy(float)
        R = s05_arcs.resample_matrix(len(x), T)
        sm = arcs[arcs.movie_name == r.movie_name].smooth.to_numpy()
        for which, k in (("arc minimum", int(sm.argmin())), ("arc maximum", int(sm.argmax()))):
            cols = np.flatnonzero(R[max(0, k - h):k + h + 1].sum(0) > 0)        # scenes under the window
            j = cols[np.argmin(x[cols])] if which == "arc minimum" else cols[np.argmax(x[cols])]
            scene = int(g.scene.iloc[j])
            ss = sents[(sents.movie_name == r.movie_name) & (sents.scene == scene)]
            s = ss.loc[ss.score.idxmin()] if which == "arc minimum" else ss.loc[ss.score.idxmax()]
            row = {"movie_name": r.movie_name, "descriptor": r.descriptor, "selection": r.side, "rank": r.rank,
                   "D1": round(r.D1, 4), "D2": round(r.D2, 4), "point": which, "story_time": round((k + 0.5) / T, 3),
                   "scene": scene, "scene_position": round(j / max(len(x) - 1, 1), 3), "sentences_in_scene": len(ss),
                   f"{inst}_scene": round(float(x[j]), 4), "sentence_pos": int(s.pos), "sentence_layer": s.layer,
                   f"{inst}_sentence": round(float(s.score), 4)}
            for i, o in others.items():
                v = o[(o.movie_name == r.movie_name) & (o.scene == scene)][f"{i}_both"]
                row[f"{i}_scene"] = round(float(v.iloc[0]), 4) if len(v) else None
            rows.append(row)
            win = _window(s.text, lex, c["back_to_text"]["window_words"])
            priv.append({**{k_: row[k_] for k_ in ("movie_name", "descriptor", "point", "scene", "sentence_pos")},
                         "window": win, "words": len(win.split())})
            sign = -1 if which == "arc minimum" else 1
            words[which].update(t.lower() for t in TOKEN.findall(s.text) if sign * lex.get(t.lower(), 0) > 0)
    sheet = pd.DataFrame(rows)
    for col in HUMAN:
        sheet[col] = ""
    sheet.to_csv(SHEET, index=False)
    wl = [pd.DataFrame(words[k].most_common(20), columns=[f"word pulling the {k}", f"count ({k})"])
          for k in ("arc minimum", "arc maximum")]
    pd.concat(wl, axis=1).to_csv(WORDS, index=False)
    PRIV.mkdir(parents=True, exist_ok=True)
    pv = pd.DataFrame(priv)
    assert pv.words.max() <= c["back_to_text"]["window_words"]
    pv.to_csv(PRIV / "windows.csv", index=False)
    page = ["<meta charset='utf-8'><style>body{font:15px/1.5 system-ui;max-width:52em;margin:2em auto}</style>",
            "<h1>Back to the text (PRIVATE: script excerpts, at most 15 words each)</h1>",
            "<p>For each row of <code>analysis/arcs/outputs/s12_back_to_text.csv</code>: does the score match the "
            f"text (yes/no), and what drives the extreme ({' / '.join(DRIVERS)})? The text wins.</p>"]
    for p in pv.itertuples():
        page.append(f"<p><b>{html.escape(p.movie_name)}</b> · {p.descriptor} · {p.point} · scene {p.scene}<br>"
                    f"{html.escape(p.window)}</p>")
    (PRIV / "windows.html").write_text("\n".join(page), encoding="utf8")
    META.write_text(json.dumps({"instrument": inst, "why": why, "rows": len(sheet), "max_window_words": int(pv.words.max())})
                    + "\n")
    log_run("s12_back_to_text", [SHEET.name, WORDS.name, META.name], {"back_to_text": c["back_to_text"], "instrument": inst})
    return load()


def load() -> dict:
    return {"sheet": pd.read_csv(SHEET, dtype={"movie_name": str}), "words": pd.read_csv(WORDS),
            "meta": json.loads(META.read_text())}


def tables(res: dict) -> dict:
    s = res["sheet"]
    inst = res["meta"]["instrument"]
    pos = s.groupby(["descriptor", "point"]).agg(rows=("scene", "size"), median_story_time=("story_time", "median"),
                                                 share_in_last_fifth=("story_time", lambda v: round(float((v >= .8).mean()), 2)),
                                                 share_in_first_fifth=("story_time", lambda v: round(float((v < .2).mean()), 2)),
                                                 median_scene_sentences=("sentences_in_scene", "median"),
                                                 action_line_drivers=("sentence_layer", lambda v: int((v == "action").sum())))
    return {"sheet": s.fillna(""), "positions": pos.reset_index(), "words": res["words"].head(20),
            "status": pd.DataFrame([("instrument", f"{inst}: {res['meta']['why']}"), ("rows", res["meta"]["rows"]),
                                    ("human columns filled", int((s[HUMAN].fillna("") != "").sum().sum())),
                                    ("longest private window (words)", res["meta"]["max_window_words"])],
                                   columns=["item", "value"])}


def private(res: dict) -> dict:
    return {"windows": pd.read_csv(PRIV / "windows.csv")}


def figures(res: dict) -> dict:
    return {}


def reading(res: dict) -> str:
    s = res["sheet"]
    w = res["words"]
    neg = s[(s.selection.str.startswith("most negative")) & (s.point == "arc minimum")]
    one = (s.sentences_in_scene == 1).mean()
    mn = s[s.point == "arc minimum"]
    top = lambda col: ", ".join(w[col].dropna().head(5))
    return (f"The {len(s)} rows cover {s.movie_name.nunique()} scripts (some are extreme on both descriptors). "
            f"{one:.0%} of the scenes at an arc extreme hold a single sentence (median "
            f"{s.sentences_in_scene.median():.0f}), and {int((mn.sentence_layer == 'action').sum())} of {len(mn)} "
            f"sentences driving an arc minimum are action lines. So the most extreme arcs are shaped by short "
            f"scenes, where one strongly scored line becomes the whole scene score. That fits the split-half "
            f"reliability of Step 5. For the {len(neg)} scripts that end darkest, the arc minimum falls in the last "
            f"fifth of the story for {int((neg.story_time >= .8).sum())} of them. The VADER-lexicon words in those sentences most often "
            f"pulling arc minima are {top('word pulling the arc minimum')}; those lifting arc maxima are "
            f"{top('word pulling the arc maximum')}. Whether these are described events (a death on the page), "
            f"artefacts or tone is for a human reader to decide from the private windows. Until then the sheet "
            f"reads NOT YET VALIDATED.")


SECTION = [
    Md("""## Step 12 · Back to the text

Numbers about arcs are only as good as what they point at in the scripts. Here we take the extremes:
- the **10 most volatile scripts** (D1);
- the **5 scripts that end darkest and the 5 that end lightest** relative to their start (D2).

For each, we find the scene at the arc's lowest and highest point, and the sentence in that scene that pushes
the score hardest. The sentence is cut to at most 15 words and kept in `private/`.

A human then checks two things:
- **Does the score match the text?**
- **What drives the extreme?** A described event (a death, a killing), a tokenisation or markup artefact, or
  genuine tone.

A lexicon scores "He is dead." as strongly negative whether the scene is a tragedy or a joke. This step finds
out which it is. **If the code and the text disagree, the text wins.**

No outcome is used here."""),
    Run("s12_back_to_text"),
    Table("s12_back_to_text", "status", "What this sheet is built from and how far the human check has got."),
    Table("s12_back_to_text", "positions", "Where the extremes sit in story time, and whether action lines or dialogue "
          "drive them."),
    Table("s12_back_to_text", "words", "Lexicon words in the driving sentences that pull an arc minimum down or lift an "
          "arc maximum (single words, counted over all rows)."),
    Table("s12_back_to_text", "sheet", "The check sheet as committed: keys, scores and empty human columns. No text."),
    Private("s12_back_to_text", "windows", "the ≤ 15-word windows (notebook only; outputs cleared before commit)."),
    Reading("s12_back_to_text"),
]
