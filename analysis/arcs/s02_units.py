"""Step 2: text units. Scenes in order, each split into dialogue and action sentences (contractions joined first)."""
from __future__ import annotations

import html
import json
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from analysis.scripts_text.report import HIGH, LOW, NEUTRAL, Fig, Md, Private, Reading, Run, Table, style

from . import s01_sample
from .common import pooled_cells
from .config import cfg, log_run
from .paths import MOVIESUM, OUTPUTS, PRIVATE

SENTENCES = PRIVATE / "sentences.parquet"    # script text: never committed
REPORT = OUTPUTS / "s02_parser_report.csv"   # per-script counts, keyed by MovieSum name
TABLE = OUTPUTS / "analysis_table.csv"       # s01 films + per-script text stats + sample flags (no IMDb fields)
SCENES = OUTPUTS / "s02_scene_counts.csv.gz"  # per-scene sentence counts (numbers only)
SUMMARY = OUTPUTS / "s02_summary.json"

SCENE = re.compile(r"<scene>(.*?)</scene>", re.S)
TAG = re.compile(r"<(stage_direction|scene_description|character|parenthetical|dialogue)>(.*?)</\1>", re.S)
SPLIT_CONTRACTION = re.compile(cfg()["text"]["split_contraction_regex"], re.I)
SENT = re.compile(cfg()["text"]["sentence_split_regex"])
WORD = re.compile(cfg()["text"]["word_regex"])
LONE_NT = re.compile(r"(?<=\s)n't\b", re.I)          # "n't" standing alone: the symptom of pre-tokenising
NUMBERED = re.compile(r"^\s*\d+[A-Z]?[.\s]")
EXAMPLES = ["Pulp Fiction_1994", "Inception_2010", "The Godfather_1972"]
LAYER_TAG = {"dialogue": "dialogue", "scene_description": "action"}


def split_sentences(body: str) -> list[str]:
    return [s.strip() for s in SENT.split(body) if WORD.search(s)]


def parse(script: str) -> tuple[list[dict], dict]:
    """Scenes in order of appearance. Each scene: its dialogue and action sentences (contractions joined
    before the split). Headings, speaker cues and parentheticals are not scored. Returns the sentence
    rows and per-script counts."""
    rows, c = [], {"n_scene_tags": 0, "headings": 0, "numbered_headings": 0, "joins": 0, "lone_nt_before": 0,
                   "lone_nt_after": 0, "tags_dialogue": 0, "tags_action": 0, "tags_other": 0,
                   "words_dialogue": 0, "words_action": 0}
    scenes = SCENE.findall(script)
    c["n_scene_tags"] = len(scenes)
    for i, block in enumerate(scenes):
        k = 0
        for tag, body in TAG.findall(block):
            body = html.unescape(body).strip()
            if not body:
                continue
            if tag == "stage_direction":
                c["headings"] += 1
                c["numbered_headings"] += bool(NUMBERED.match(body))
            if tag not in LAYER_TAG:
                c["tags_other"] += 1
                continue
            layer = LAYER_TAG[tag]
            c[f"tags_{layer}"] += 1
            c["lone_nt_before"] += len(LONE_NT.findall(body))
            c["joins"] += len(SPLIT_CONTRACTION.findall(body))
            body = SPLIT_CONTRACTION.sub(r"\1\2", body)
            c["lone_nt_after"] += len(LONE_NT.findall(body))
            for s in split_sentences(body):
                n = len(WORD.findall(s))
                c[f"words_{layer}"] += n
                rows.append({"scene": i, "layer": layer, "pos": k, "words": n, "text": s})
                k += 1
    return rows, c


def _scripts(names: set[str]) -> dict[str, str]:
    """Longest MovieSum script per film, by MovieSum name."""
    best = {}
    for path in MOVIESUM:
        with open(path, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if r["movie_name"] in names and len(r["script"] or "") > len(best.get(r["movie_name"], "")):
                    best[r["movie_name"]] = r["script"] or ""
    return best


def run() -> dict:
    c = cfg()
    films = s01_sample.load()["films"]
    kept = films[~films.excluded_dialogue_markup]
    scripts = _scripts(set(kept.movie_name))
    sent_rows, rep = [], []
    for name in kept.movie_name:
        try:
            rows, cnt = parse(scripts[name])
            err = None
        except Exception as e:     # counted, never silently dropped
            rows, cnt, err = [], {}, repr(e)
        for r in rows:
            r["movie_name"] = name
        sent_rows += rows
        rep.append({"movie_name": name, "parse_error": err, **cnt})
    sents = pd.DataFrame(sent_rows)[["movie_name", "scene", "layer", "pos", "words", "text"]]
    sents.to_parquet(SENTENCES, index=False)

    # Per scene: sentences per layer (numbers only).
    sc = sents.groupby(["movie_name", "scene", "layer"]).size().unstack("layer", fill_value=0).reset_index()
    sc["both"] = sc.dialogue + sc.action
    sc.to_csv(SCENES, index=False, compression="gzip")

    rep = pd.DataFrame(rep)
    per = sc.groupby("movie_name").agg(scenes_nonempty_both=("both", lambda v: int((v > 0).sum())),
                                       scenes_nonempty_dialogue=("dialogue", lambda v: int((v > 0).sum())),
                                       scenes_nonempty_action=("action", lambda v: int((v > 0).sum())),
                                       sentences_dialogue=("dialogue", "sum"), sentences_action=("action", "sum"),
                                       mean_sentences_per_scene=("both", lambda v: float(v[v > 0].mean())))
    rep = rep.merge(per, on="movie_name", how="left")
    rep["sentences_both"] = rep.sentences_dialogue + rep.sentences_action
    rep["empty_scenes_both"] = rep.n_scene_tags - rep.scenes_nonempty_both
    rep["action_share"] = rep.words_action / (rep.words_action + rep.words_dialogue)
    rep["possible_transcript"] = (rep.action_share < 0.15) | (rep.n_scene_tags < c["sample"]["arc_ready"]["min_scene_tags"])
    rep["pretokenised"] = rep.joins >= 20
    rep["numbered_scenes"] = (rep.headings > 0) & (rep.numbered_headings >= 0.5 * rep.headings)
    ar = c["sample"]["arc_ready"]

    def ready(r, rule):
        return (r.n_scene_tags >= rule["min_scene_tags"]) & (r.sentences_both >= rule["min_sentences"])
    rep["arc_ready"] = ready(rep, ar)
    rep["arc_ready_strict"] = ready(rep, ar["sensitivity_strict"])
    rep["arc_ready_loose"] = ready(rep, ar["sensitivity_loose"])
    rep.to_csv(REPORT, index=False)

    chk = kept.set_index("movie_name").n_scene_tags.reindex(rep.movie_name).to_numpy()
    assert (chk == rep.n_scene_tags.to_numpy()).all(), "scene-tag counts differ between Step 1 and Step 2"
    t = kept.merge(rep.drop(columns=["n_scene_tags"]), on="movie_name", how="left")
    t["in_primary"] = ~t.likely_franchise & t.arc_ready
    t["cell"] = pooled_cells(t.primary_genre, t.decade_bin)   # for all scripts; steps re-pool on their own sample
    t.to_csv(TABLE, index=False)

    nf = t[~t.likely_franchise]
    ex = rep.set_index("movie_name")
    summary = {
        "scripts": int(len(rep)), "parse_errors": int(rep.parse_error.notna().sum()),
        "sentences": {l: int(sents.layer.eq(l).sum()) for l in ("dialogue", "action")} | {"both": int(len(sents))},
        "scenes_total": int(rep.n_scene_tags.sum()), "empty_scenes_both": int(rep.empty_scenes_both.sum()),
        "contraction_joins": int(rep.joins.sum()), "lone_nt_before": int(rep.lone_nt_before.sum()),
        "lone_nt_after": int(rep.lone_nt_after.sum()), "pretokenised": int(rep.pretokenised.sum()),
        "possible_transcripts": int(rep.possible_transcript.sum()), "numbered_scenes": int(rep.numbered_scenes.sum()),
        "arc_ready_all": int(rep.arc_ready.sum()), "arc_ready_primary": int(nf.arc_ready.sum()),
        "arc_ready_strict_primary": int(nf.arc_ready_strict.sum()), "arc_ready_loose_primary": int(nf.arc_ready_loose.sum()),
        "excluded_primary": {"under_min_scenes": int((nf.n_scene_tags < ar["min_scene_tags"]).sum()),
                             "under_min_sentences_only": int(((nf.n_scene_tags >= ar["min_scene_tags"]) &
                                                              (nf.sentences_both < ar["min_sentences"])).sum())},
        "possible_transcripts_in_primary": int(t.in_primary[t.possible_transcript].sum()),
        "pretokenised_in_primary": int(t.in_primary[t.pretokenised].sum()),
        "median": {k: float(t.loc[t.in_primary, k].median()) for k in
                   ("n_scene_tags", "sentences_both", "mean_sentences_per_scene", "action_share")},
        "examples": {n: {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                         for k, v in ex.loc[n].items()} for n in EXAMPLES if n in ex.index},
    }
    if summary["arc_ready_primary"] != c["sample"]["expected"]["not_likely_franchise_arc_ready"]:
        raise RuntimeError(f"arc-ready primary {summary['arc_ready_primary']} != expected 834")
    SUMMARY.write_text(json.dumps(summary, indent=1, default=str) + "\n")
    log_run("s02_units", [REPORT.name, TABLE.name, SCENES.name, SUMMARY.name], {"text": c["text"], "arc_ready": ar})
    return load()


def load() -> dict:
    return {"report": pd.read_csv(REPORT), "table": pd.read_csv(TABLE),
            "scenes": pd.read_csv(SCENES), "summary": json.loads(SUMMARY.read_text())}


def load_sentences() -> pd.DataFrame:
    return pd.read_parquet(SENTENCES)


# ---------------------------------------------------------------- tables

def _sanity(res: dict) -> pd.DataFrame:
    s, rep = res["summary"], res["report"]
    rows = [("scripts parsed", s["scripts"], ""), ("parse errors", s["parse_errors"], ""),
            ("scene tags in all scripts", s["scenes_total"], ""),
            ("scenes with no dialogue or action sentence (dropped from the timeline)", s["empty_scenes_both"],
             "mostly a heading or a speaker cue alone"),
            ("sentences, dialogue", s["sentences"]["dialogue"], ""), ("sentences, action", s["sentences"]["action"], ""),
            ("split contractions joined", s["contraction_joins"], "\"do n't\" → \"don't\""),
            ("pre-tokenised scripts (≥ 20 joins)", s["pretokenised"], "fix applied to every script anyway"),
            ("possible transcripts (action share < 0.15 or < 20 scene tags)", s["possible_transcripts"],
             f"{s['possible_transcripts_in_primary']} of them in the primary sample; flagged, kept"),
            ("scripts whose headings are mostly numbered", s["numbered_scenes"],
             "the rest give no scene numbers: order of appearance is the timeline")]
    return pd.DataFrame(rows, columns=["check", "count", "note"])


def _funnel(res: dict) -> pd.DataFrame:
    s = res["summary"]
    e = s["excluded_primary"]
    ar = cfg()["sample"]["arc_ready"]
    return pd.DataFrame([
        {"stage": "not likely franchise (Step 1)", "films": cfg()["sample"]["expected"]["not_likely_franchise"], "note": ""},
        {"stage": f"… with fewer than {ar['min_scene_tags']} scene tags", "films": -e["under_min_scenes"],
         "note": "scene markup missing: the film would read as a handful of scenes"},
        {"stage": f"… with enough scenes but fewer than {ar['min_sentences']} sentences", "films": -e["under_min_sentences_only"],
         "note": "too little text for a 100-point arc"},
        {"stage": "PRIMARY SAMPLE: arc-ready, not likely franchise", "films": s["arc_ready_primary"], "note": "expected 834"},
        {"stage": "sensitivity: strict filter (≥ 40 scenes, ≥ 600 sentences)", "films": s["arc_ready_strict_primary"], "note": ""},
        {"stage": "sensitivity: loose filter (≥ 10 scenes, ≥ 150 sentences)", "films": s["arc_ready_loose_primary"], "note": ""},
        {"stage": "all scripts, arc-ready (franchise included)", "films": s["arc_ready_all"], "note": "sensitivity sample"},
    ])


def _examples(res: dict) -> pd.DataFrame:
    """Before/after as counts: what the markup holds against what the units keep. No text."""
    rows = []
    for name, e in res["summary"]["examples"].items():
        rows.append({"film": name, "scene tags": e["n_scene_tags"], "headings (not scored)": e["headings"],
                     "dialogue tags": e["tags_dialogue"], "description tags": e["tags_action"],
                     "cues + parentheticals (not scored)": e["tags_other"] - e["headings"],
                     "scenes kept on the timeline": e["scenes_nonempty_both"],
                     "dialogue sentences": e["sentences_dialogue"], "action sentences": e["sentences_action"],
                     "mean sentences / scene": round(e["mean_sentences_per_scene"], 1),
                     "split contractions joined": e["joins"]})
    return pd.DataFrame(rows)


def _contractions(res: dict) -> pd.DataFrame:
    s = res["summary"]
    return pd.DataFrame([
        {"what": "standalone \"n't\" tokens (a negation cut off from its verb)", "before the fix": s["lone_nt_before"],
         "after the fix": s["lone_nt_after"]},
        {"what": "example (constructed, not from a script)", "before the fix": "I do n't love you .",
         "after the fix": "I don't love you ."},
        {"what": "does it change a VADER score?", "before the fix": "no, on the hand-made cases: VADER's negation "
         "check matches any token containing \"n't\", so a split \"n't\" still negates",
         "after the fix": "measured on every affected sentence in Step 3"},
        {"what": "why we fix it anyway", "before the fix": "\"n't\" is counted as a word; \"ca\", \"wo\" appear as words; "
         "a contextual model sees an unusual token sequence", "after the fix": "word counts and tokenisation are the same "
         "for pre-tokenised and ordinary scripts"}])


def tables(res: dict) -> dict:
    t = res["table"]
    p = t[t.in_primary]
    desc = p[["n_scene_tags", "scenes_nonempty_both", "sentences_both", "sentences_dialogue", "sentences_action",
              "mean_sentences_per_scene", "action_share"]].describe(percentiles=[.1, .5, .9]).T.round(2)
    desc.index.name = "per script (primary sample)"
    return {"sanity": _sanity(res), "funnel": _funnel(res), "examples": _examples(res),
            "contractions": _contractions(res), "describe": desc.reset_index()}


def private(res: dict) -> dict:
    """Up to 5 real before/after negation cases, each cut to 15 words. Shown in the notebook only."""
    s = load_sentences()
    hit = s[s.text.str.contains(r"\b\w+n't\b", regex=True) & (s.words <= 15)].sample(5, random_state=cfg()["seed"])
    return {"negation": hit[["movie_name", "scene", "layer", "text"]].assign(text=hit.text.str.split().str[:15].str.join(" "))}


# ---------------------------------------------------------------- figures

def figures(res: dict) -> dict:
    t = res["table"]
    nf = t[~t.likely_franchise]
    ar = cfg()["sample"]["arc_ready"]
    sc = go.Figure()
    for flag, colour, sym, name in ((True, HIGH, "circle", "arc-ready"), (False, LOW, "x", "excluded")):
        v = nf[nf.arc_ready == flag]
        sc.add_trace(go.Scattergl(x=v.n_scene_tags, y=v.sentences_both, mode="markers", name=f"{name} (n={len(v)})",
                                  marker=dict(color=colour, symbol=sym, size=6, opacity=0.6)))
    sc.add_vline(x=ar["min_scene_tags"], line_dash="dash", annotation_text=f"{ar['min_scene_tags']} scenes")
    sc.add_hline(y=ar["min_sentences"], line_dash="dash", annotation_text=f"{ar['min_sentences']} sentences")
    sc.update_xaxes(type="log", title="scene tags per script (log scale)")
    sc.update_yaxes(type="log", title="dialogue + action sentences per script (log scale)")
    style(sc, "Which non-franchise scripts are long and well marked up enough for an arc", height=420)

    p = t[t.in_primary]
    sps = go.Figure(go.Histogram(x=p.mean_sentences_per_scene, nbinsx=50, marker_color=HIGH))
    sps.update_layout(xaxis_title="mean sentences per non-empty scene (scene length)", yaxis_title="scripts",
                      showlegend=False)
    style(sps, "Scene length varies a lot between scripts: it is a control", height=340)

    scn = res["scenes"]
    scn = scn[scn.movie_name.isin(p.movie_name) & (scn.both > 0)]
    per = go.Figure(go.Histogram(x=scn.both.clip(upper=80), nbinsx=80, marker_color=NEUTRAL))
    per.update_layout(xaxis_title="sentences in one scene (dialogue + action; 80 = 80 or more)",
                      yaxis_title="scenes", showlegend=False)
    style(per, "Most scenes are short: a scene score is a mean over few sentences", height=340)
    return {"arc_ready": sc, "scene_length": sps, "sentences_per_scene": per}


# ---------------------------------------------------------------- reading

def reading(res: dict) -> str:
    s = res["summary"]
    m = s["median"]
    return (
        f"All {s['scripts']:,} scripts parse ({s['parse_errors']} errors). They hold {s['scenes_total']:,} scene tags; "
        f"{s['empty_scenes_both']:,} scenes have no dialogue or description sentence and drop out of the timeline. "
        f"Joining {s['contraction_joins']:,} split contractions cut the standalone \"n't\" tokens from "
        f"{s['lone_nt_before']:,} to {s['lone_nt_after']:,}; {s['pretokenised']} scripts are pre-tokenised (≥ 20 joins). "
        f"{s['arc_ready_primary']} non-franchise scripts are arc-ready (the expected 834): the median one has "
        f"{m['n_scene_tags']:.0f} scenes, {m['sentences_both']:,.0f} sentences and {m['mean_sentences_per_scene']:.1f} "
        f"sentences per scene, with {m['action_share']:.0%} of its words in descriptions. "
        f"{s['possible_transcripts_in_primary']} possible transcripts remain in the primary sample (flagged; Step 10 "
        f"drops them). Only {s['numbered_scenes']} scripts number most of their scene headings, so for nearly all "
        f"scripts the order in which scenes appear in the file is taken to be the story order: an assumption we "
        f"cannot check from the markup."
    )


SECTION = [
    Md("""## Step 2 · Text units: scenes and sentences

**What we do.** We cut each screenplay into its `<scene>` blocks, in the order they appear, and split each
scene into two layers:
- **dialogue**: what characters say, from the `<dialogue>` tags;
- **action**: what the script describes happening, from the `<scene_description>` tags.

Their union, **both**, is the primary layer. Scene headings ("INT. KITCHEN - NIGHT"), speaker names and
parentheticals are not scored: they carry places and names, not tone.

Inside each tag we split the text into sentences after `.`, `!` or `?`. A tag boundary is always a sentence
boundary. Sentiment is later computed **per sentence** and averaged per scene.

**Before splitting we fix split contractions.** Some MovieSum scripts are pre-tokenised: "don't" appears as
"do n't". We expected this to break negation in a lexicon. Testing showed that VADER's negation check catches
a split "n't" anyway, so for VADER the fix is a safeguard, not a correction. Step 3 measures this on every
affected sentence. The fix still matters for word counts (otherwise "n't", "ca" and "wo" count as words) and
for a contextual model, whose tokenizer would see an unusual sequence. Pre-tokenised and ordinary scripts are
now processed the same way.

**Why the scene is the unit of time.**
- A scene is the screenwriter's own unit of story: one place, one stretch of time.
- Pages or words would make long description passages count more than quick exchanges.
- Speeches would make talky scenes count more than action scenes.

**Assumption.** The scene order in the file is the story order. Only a handful of scripts number their scenes,
so we can't check it.

**What it misses.**
- Scenes differ in length, so a short scene's mean rests on few sentences and is noisy (Step 5 checks this).
- A scene with only a heading has no sentences and drops out of the timeline. It is counted, not hidden.
- Sentence splitting on punctuation misses sentences without end punctuation and splits at "Mr.", which only
  changes how finely a scene is cut, not what it says."""),
    Run("s02_units"),
    Table("s02_units", "sanity", "Parser sanity report over the 1,087 scripts left after Step 1's exclusions."),
    Table("s02_units", "contractions", "Before/after for the contraction fix, counted over every script, with a "
          "constructed example so no script text is needed."),
    Private("s02_units", "negation", "five real before/after negation cases, at most 15 words each (notebook only)."),
    Table("s02_units", "examples", "Before/after for three example scripts, as counts only: what the markup holds "
          "and what becomes a scored unit. Headings, speaker cues and parentheticals are kept out of the score."),
    Fig("s02_units", "arc_ready", "Each mark is a non-franchise script: scene tags (x) against dialogue + action "
        "sentences (y), both on log scales. Dashed lines are the pre-registered arc-ready filter. Excluded scripts "
        "(orange crosses) sit left of or below the lines: their scene markup is missing or they are too short for "
        "a 100-point arc."),
    Table("s02_units", "funnel", "The funnel continued from Step 1: the arc-ready filter, its two sensitivity "
          "versions, and the all-scripts sample used in Steps 9 and 10."),
    Table("s02_units", "describe", "Per-script size in the primary sample: 10th, 50th and 90th percentiles."),
    Fig("s02_units", "scene_length", "Mean sentences per scene for each primary script. This is the 'scene length' "
        "control: a script of short scenes gives noisier scene scores, which could inflate volatility (D1)."),
    Fig("s02_units", "sentences_per_scene", "Sentences in a single scene, over all scenes of the primary scripts. "
        "Many scenes have only a few sentences, which is why scene scores are noisy and the arc is smoothed."),
    Reading("s02_units"),
]
