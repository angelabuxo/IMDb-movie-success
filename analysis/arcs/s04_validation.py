"""Step 4: human validation sheets. The agent draws and prepares; humans fill. No column for humans is ever
filled here, and an existing sheet is never redrawn (only its instrument columns are refreshed).

    python -m analysis.arcs.s04_validation          # draw the sheets (first time) or refresh instrument columns
    python -m analysis.arcs.s04_validation score    # once filled: accuracy, kappa, error table

Committed:  analysis/arcs/validation/scenes_sheet.csv, scripts_sheet.csv (MovieSum names, derived genre bin,
            numbers, empty human columns), outputs/s04_validation_results.json.
Private:    private/validation/scenes_preview.html, scripts_preview.html (script text and IMDb title/year,
            for the raters only).
"""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from analysis.scripts_text.report import Md, Private, Reading, Run, Table

from . import s02_units, s03_instruments
from .config import cfg, log_run
from .paths import OUTPUTS, PRIVATE, REPO

VAL = REPO / "analysis" / "arcs" / "validation"
SCENES = VAL / "scenes_sheet.csv"
SCRIPTS = VAL / "scripts_sheet.csv"
PREVIEW = PRIVATE / "validation"
RESULTS = OUTPUTS / "s04_validation_results.json"
INSTRUMENTS = ("vader", "sst2", "roberta3")
SCENE_HUMAN = ["rater1_label", "rater1_error", "rater2_label", "rater2_error", "adjudicated_label", "notes"]
SCRIPT_HUMAN = ["markup_ok", "kind", "id_mapping_ok", "rater", "notes"]
NOT_VALIDATED = "NOT YET VALIDATED: the sheets have not been filled"


# ---------------------------------------------------------------- draw

def _allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional, largest remainder, at least 1 per bin."""
    raw = sizes / sizes.sum() * n
    k = np.maximum(np.floor(raw).astype(int), 1)
    while k.sum() < n:
        k[(raw - k).idxmax()] += 1
    while k.sum() > n:
        k[(k - raw).where(k > 1).idxmax()] -= 1
    return k


def draw() -> None:
    c = cfg()
    v = c["validation"]
    rng = np.random.default_rng(c["seed"])
    tab = s02_units.load()["table"]
    prim = tab[tab.in_primary][["movie_name", "primary_genre", "decade_bin"]]
    sc = s03_instruments.load_scores("vader", "scene").merge(prim, on="movie_name")
    k = _allocate(sc.groupby("primary_genre").size(), v["scenes"])
    parts = [g.iloc[rng.choice(len(g), k[name], replace=False)] for name, g in sc.groupby("primary_genre")]
    s = pd.concat(parts).sample(frac=1, random_state=c["seed"]).reset_index(drop=True)
    s.insert(0, "sheet_id", [f"S{i + 1:02d}" for i in range(len(s))])
    s = s[["sheet_id", "movie_name", "scene", "primary_genre", "n_both", "n_dialogue", "n_action"]].rename(
        columns={"primary_genre": "genre_bin", "n_both": "sentences"})
    for col in SCENE_HUMAN:
        s[col] = ""
    VAL.mkdir(parents=True, exist_ok=True)
    s.to_csv(SCENES, index=False)

    kept = tab[~tab.excluded_dialogue_markup]
    d = kept.iloc[rng.choice(len(kept), v["scripts"], replace=False)]
    d = d[["movie_name", "primary_genre", "n_scene_tags", "sentences_both", "action_share", "pretokenised",
           "possible_transcript", "numbered_scenes", "in_primary"]].rename(columns={"primary_genre": "genre_bin"})
    d.insert(0, "sheet_id", [f"P{i + 1:02d}" for i in range(len(d))])
    d["action_share"] = d.action_share.round(3)
    for col in SCRIPT_HUMAN:
        d[col] = ""
    d.to_csv(SCRIPTS, index=False)


def refresh_scores() -> pd.DataFrame:
    """Add or update instrument columns from Step 3's saved scene scores. Human columns are checked unchanged."""
    s = pd.read_csv(SCENES, dtype=str, keep_default_na=False)
    before = s[SCENE_HUMAN].copy()
    for inst in INSTRUMENTS:
        col = f"{inst}_scene"
        try:
            sc = s03_instruments.load_scores(inst, "scene")[["movie_name", "scene", f"{inst}_both"]]
        except FileNotFoundError:
            s[col] = s.get(col, "")
            continue
        sc["scene"] = sc.scene.astype(str)
        m = s[["movie_name", "scene"]].merge(sc, on=["movie_name", "scene"], how="left", validate="one_to_one")
        s[col] = m[f"{inst}_both"].round(4).astype(str).replace("nan", "")
    for key in ("sst2", "roberta3"):   # scene-in-context configuration (Step 9 robustness), when present
        p = OUTPUTS / f"s03_scene_context_{key}.csv.gz"
        col = f"{key}_scene_context"
        if p.exists():
            sc = pd.read_csv(p, dtype={"movie_name": str})[["movie_name", "scene", f"{key}_scene"]]
            sc["scene"] = sc.scene.astype(str)
            m = s[["movie_name", "scene"]].merge(sc, on=["movie_name", "scene"], how="left", validate="one_to_one")
            s[col] = m[f"{key}_scene"].round(4).astype(str).replace("nan", "")
        else:
            s[col] = s.get(col, "")
    assert s[SCENE_HUMAN].equals(before), "human columns changed"
    order = [c for c in s.columns if c not in SCENE_HUMAN] + SCENE_HUMAN
    s[order].to_csv(SCENES, index=False)
    return s[order]


def previews() -> None:
    """Private, gitignored: what the raters read. Scene text without any score; script openings with IMDb title."""
    c = cfg()["validation"]
    PREVIEW.mkdir(parents=True, exist_ok=True)
    sents = s02_units.load_sentences()
    s = pd.read_csv(SCENES, dtype={"movie_name": str})
    head = ("<meta charset='utf-8'><style>body{font:15px/1.5 system-ui;max-width:52em;margin:2em auto;padding:0 1em}"
            "h2{margin-top:2.5em;border-top:1px solid #999;padding-top:.6em}.a{color:#555}.d{margin-left:2em}</style>")
    out = [head, "<h1>Scene sheet: rater preview (PRIVATE: screenplay text, do not share or commit)</h1>",
           "<p>For each scene, write in <code>analysis/arcs/validation/scenes_sheet.csv</code>, in your rater "
           f"columns: the overall tone of the scene ({' / '.join(c['labels'])}), and, if you think a sentiment "
           f"tool would get it wrong, why ({' / '.join(c['error_categories'])}). Judge the tone the scene conveys "
           "to a reader, not whether its events are good for the hero. No instrument score is shown here on "
           "purpose: do not look at the score columns of the sheet before labelling. Rater 2 labels without "
           "seeing rater 1's labels.</p>"]
    for r in s.itertuples():
        x = sents[(sents.movie_name == r.movie_name) & (sents.scene == r.scene)].sort_values("pos")
        out.append(f"<h2>{r.sheet_id}</h2><p><i>{html.escape(r.genre_bin)} · {len(x)} sentences</i></p>")
        out += [f"<p class='{'a' if l == 'action' else 'd'}'>{html.escape(t)}</p>" for l, t in zip(x.layer, x.text)]
    (PREVIEW / "scenes_preview.html").write_text("\n".join(out), encoding="utf8")

    films = pd.read_parquet(PRIVATE / "films.parquet")[["movie_name", "title_imdb", "year", "tconst"]]
    p = pd.read_csv(SCRIPTS, dtype={"movie_name": str}).merge(films, on="movie_name", how="left")
    out = [head, "<h1>Script sheet: rater preview (PRIVATE: screenplay text and IMDb fields)</h1>",
           "<p>For each script, fill in <code>scripts_sheet.csv</code>: <b>markup_ok</b> (yes/no: are scenes, "
           "dialogue and descriptions tagged as such below?), <b>kind</b> (final / draft / transcript / unclear), "
           "<b>id_mapping_ok</b> (yes/no: is this the screenplay of the IMDb film named here?), <b>rater</b>, "
           "<b>notes</b>. The first three and the last scene are shown, with D = dialogue and A = description.</p>"]
    for r in p.itertuples():
        x = sents[sents.movie_name == r.movie_name]
        keep = sorted(x.scene.unique())
        keep = keep[:3] + keep[-1:]
        out.append(f"<h2>{r.sheet_id} · MovieSum: {html.escape(r.movie_name)} → IMDb: {html.escape(str(r.title_imdb))} "
                   f"({r.year}, {r.tconst})</h2><p><i>{r.n_scene_tags} scene tags · {r.sentences_both} sentences · "
                   f"action share {r.action_share} · flagged transcript: {r.possible_transcript}</i></p>")
        for sc in keep:
            out.append(f"<h3>scene {sc}</h3>")
            y = x[x.scene == sc].sort_values("pos")
            out += [f"<p class='{'a' if l == 'action' else 'd'}'>{'A' if l == 'action' else 'D'}: {html.escape(t)}</p>"
                    for l, t in zip(y.layer, y.text)]
    (PREVIEW / "scripts_preview.html").write_text("\n".join(out), encoding="utf8")


def run() -> dict:
    if not SCENES.exists():
        draw()
    refresh_scores()
    previews()
    res = score()
    log_run("s04_validation", [SCENES.name, SCRIPTS.name, RESULTS.name], {"validation": cfg()["validation"],
                                                                          "seed": cfg()["seed"]})
    return res


# ---------------------------------------------------------------- score (after humans fill the sheets)

def _band(v: pd.Series) -> pd.Series:
    x = pd.to_numeric(v, errors="coerce")
    return pd.Series(np.where(x.isna(), "", np.where(x.abs() < 0.05, "neutral", np.where(x > 0, "positive", "negative"))),
                     index=v.index)


def score() -> dict:
    """Pre-registered scoring (config validation.scoring). Writes outputs/s04_validation_results.json."""
    c = cfg()["validation"]
    s = pd.read_csv(SCENES, dtype=str, keep_default_na=False)
    p = pd.read_csv(SCRIPTS, dtype=str, keep_default_na=False)
    for col in ("rater1_label", "rater2_label", "adjudicated_label"):
        s[col] = s[col].str.strip().str.lower()
    bad = sorted(set(s[["rater1_label", "rater2_label", "adjudicated_label"]].stack()) - set(c["labels"]) - {""})
    if bad:
        raise ValueError(f"labels outside {c['labels']}: {bad}")
    r1, r2, adj = s.rater1_label, s.rater2_label, s.adjudicated_label
    ref = np.where(adj != "", adj, np.where((r1 != "") & ((r2 == "") | (r2 == r1)), r1, ""))
    s["reference"] = ref
    unresolved = int(((r1 != "") & (r2 != "") & (r1 != r2) & (adj == "")).sum())
    res = {"scenes_in_sheet": int(len(s)), "scenes_labelled_rater1": int((r1 != "").sum()),
           "scenes_labelled_rater2": int((r2 != "").sum()), "scenes_resolved": int((s.reference != "").sum()),
           "scenes_unresolved": unresolved, "scripts_in_sheet": int(len(p)),
           "scripts_checked": int((p.markup_ok.str.strip() != "").sum())}
    res["status"] = "validated" if res["scenes_resolved"] else "not yet validated"
    if res["scenes_resolved"]:
        lab = s[s.reference != ""]
        acc = []
        for inst in [i for i in INSTRUMENTS if f"{i}_scene" in s and (s[f"{i}_scene"] != "").any()]:
            pred = _band(lab[f"{inst}_scene"])
            nm = lab.reference != "mixed"
            acc.append({"instrument": inst, "resolved scenes": int(len(lab)),
                        "accuracy (mixed counts as miss)": float((pred == lab.reference).mean()),
                        "non-mixed scenes": int(nm.sum()),
                        "accuracy on non-mixed": float((pred[nm] == lab.reference[nm]).mean()) if nm.any() else None})
        res["accuracy"] = acc
        res["reference_labels"] = lab.reference.value_counts().to_dict()
        both = (r1 != "") & (r2 != "")
        res["rater_kappa"] = float(cohen_kappa_score(r1[both], r2[both])) if both.sum() >= 2 else None
        res["rater_pairs"] = int(both.sum())
        err = pd.concat([s.rater1_error, s.rater2_error]).str.strip().str.lower()
        err = err[err != ""]
        primary = json.loads((OUTPUTS / "s03_instrument_decision.json").read_text())["primary_instrument"] \
            if (OUTPUTS / "s03_instrument_decision.json").exists() else "vader"
        wrong = lab[_band(lab[f"{primary}_scene"]) != lab.reference] if f"{primary}_scene" in lab else lab.iloc[:0]
        werr = pd.concat([wrong.rater1_error, wrong.rater2_error]).str.strip().str.lower()
        res["error_table"] = [{"category": k, "all scenes": int((err == k).sum()),
                               f"scenes {primary} got wrong": int((werr == k).sum())}
                              for k in c["error_categories"]]
        res["error_primary_instrument"] = primary
    if res["scripts_checked"]:
        q = p[p.markup_ok.str.strip() != ""]
        yes = lambda col: int(q[col].str.strip().str.lower().isin(["yes", "y"]).sum())
        res["scripts"] = {"checked": int(len(q)), "markup_ok": yes("markup_ok"), "id_mapping_ok": yes("id_mapping_ok"),
                          "kind": q.kind.str.strip().str.lower().value_counts().to_dict()}
    RESULTS.write_text(json.dumps(res, indent=1) + "\n")
    return load()


def load() -> dict:
    return {"results": json.loads(RESULTS.read_text()), "scenes": pd.read_csv(SCENES, dtype={"movie_name": str}),
            "scripts": pd.read_csv(SCRIPTS, dtype={"movie_name": str})}


# ---------------------------------------------------------------- report

def tables(res: dict) -> dict:
    r = res["results"]
    s, p = res["scenes"], res["scripts"]
    alloc = s.groupby("genre_bin").size().rename("scenes drawn").to_frame()
    tab = s02_units.load()["table"]
    sc = s03_instruments.load_scores("vader", "scene").merge(tab[tab.in_primary][["movie_name", "primary_genre"]],
                                                             on="movie_name")
    alloc["share of primary scenes"] = (sc.primary_genre.value_counts(normalize=True) * 100).round(1).astype(str) + "%"
    status = pd.DataFrame([(k, v) for k, v in r.items() if not isinstance(v, (list, dict))], columns=["item", "value"])
    out = {"status": status, "allocation": alloc.reset_index(),
           "scene_sheet": s.drop(columns=["notes"]).fillna(""), "script_sheet": p.drop(columns=["notes"]).fillna("")}
    if r["status"] == "validated":
        out["accuracy"] = pd.DataFrame(r["accuracy"]).round(3)
        out["errors"] = pd.DataFrame(r["error_table"])
    else:
        out["accuracy"] = out["errors"] = pd.DataFrame([{"result": NOT_VALIDATED}])
    return out


def private(res: dict) -> dict:
    return {"preview": pd.DataFrame([{"file": str(f.relative_to(REPO)), "bytes": f.stat().st_size}
                                     for f in sorted(PREVIEW.glob("*.html"))])}


def figures(res: dict) -> dict:
    return {}


def reading(res: dict) -> str:
    r = res["results"]
    s = res["scenes"]
    base = (f"The scene sheet holds {r['scenes_in_sheet']} scenes from the primary sample, drawn in proportion to "
            f"each genre bin ({s.genre_bin.nunique()} bins, {s.groupby('genre_bin').size().min()} to "
            f"{s.groupby('genre_bin').size().max()} scenes each); the script sheet holds {r['scripts_in_sheet']} "
            f"scripts. Scenes range from {s.sentences.min()} to {s.sentences.max()} sentences "
            f"(median {s.sentences.median():.0f}). ")
    if r["status"] != "validated":
        return base + (f"**{NOT_VALIDATED}.** Every human column is empty. Until it is filled, no instrument has been "
                       f"checked against a human reader, and every result built on the sentiment scores carries "
                       f"that caveat.")
    a = {x["instrument"]: x for x in r["accuracy"]}
    return base + "Against the human reference: " + "; ".join(
        f"{k} {v['accuracy (mixed counts as miss)']:.0%} ({v['resolved scenes']} scenes)" for k, v in a.items()) + (
        f". Agreement between raters: Cohen's kappa {r['rater_kappa']:.2f} on {r['rater_pairs']} scenes."
        if r.get("rater_kappa") is not None else ". Only one rater so far: no between-rater agreement.")


SECTION = [
    Md("""## Step 4 · Human validation sheets

**What we do.** We draw two sheets for people to fill in. The code never fills them in.
- **50 scenes** from the primary sample, drawn at random in proportion to each genre bin. Each row carries the
  instrument scene scores and empty columns for two raters: a label (positive / negative / neutral / mixed)
  and, where a tool would go wrong, an error category (negation, sarcasm, event-not-tone, tokenisation/markup,
  other).
- **25 scripts** drawn at random from all 1,087. Each row has empty columns: is the markup right, is it a final
  script, a draft or a transcript, and does the MovieSum script belong to the IMDb film we joined it to.

**Labelling is blind.** Raters read a private preview with the scene text and no score, so a number can't
anchor their judgement. The preview and the IMDb titles stay in `private/` and are never committed.

**Scoring is fixed in advance** (`validation.scoring` in the config):
- Every instrument's scene score maps to a label with the same ±0.05 band.
- Accuracy is reported with "mixed" counted as a miss, and on non-mixed scenes only.
- Cohen's kappa measures agreement between the two raters.
- The error table counts each category, overall and among the primary instrument's misses.

**What this validation can and cannot show.** 50 scenes is a small sample: an accuracy of 70% has a 95%
interval of roughly ±13 points. That is enough to show whether an instrument is usable or badly off, and not
enough to rank two instruments that are close. A scene label is also coarser than the arc, which averages many
scenes."""),
    Run("s04_validation"),
    Table("s04_validation", "status", "Sheet status. Re-running this step refreshes only the score columns and "
          "recomputes these numbers. It never redraws the sample or touches a human column."),
    Table("s04_validation", "allocation", "Scenes drawn per genre bin, against each bin's share of all scenes on "
          "the primary timeline."),
    Table("s04_validation", "scene_sheet", "The scene sheet as committed: MovieSum name, scene index, size, "
          "instrument scores and the empty rater columns. The contextual columns stay empty until the HPC job "
          "returns."),
    Table("s04_validation", "script_sheet", "The script sheet as committed, with the Step 2 flags beside the empty "
          "check columns."),
    Private("s04_validation", "preview", "where the private rater previews were written (text stays on disk)."),
    Table("s04_validation", "accuracy", "Per-instrument accuracy against the human reference label."),
    Table("s04_validation", "errors", "Error categories given by the raters."),
    Reading("s04_validation"),
]


if __name__ == "__main__":
    score() if sys.argv[1:] == ["score"] else run()
