"""Step 11: human validation sheets. The agent prepares them; people fill them in.

Committed sheets (outputs/validation_*.csv) hold IDs, numbers and EMPTY columns. The text a person needs to
judge (short previews, KWIC lines) is written next to them in private/, never committed.

After the team fills the empty columns, `python -m analysis.scripts_text.s11_validation score` (or rerunning the
notebook) computes agreement and precision per category and writes outputs/validation_results.json, which the
notebook and the page then show instead of "NOT YET VALIDATED".
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np
import pandas as pd

from . import s01_groups, s02_layers, s05_scatter, s08_kwic
from .config import cfg, log_run
from .paths import OUTPUTS, PRIVATE
from .report import Md, Reading, Run, Table

SCRIPTS_SHEET = OUTPUTS / "validation_scripts.csv"
TERMS_SHEET = OUTPUTS / "validation_terms.csv"
SCRIPTS_PREVIEW = PRIVATE / "validation_scripts_preview.csv"
TERMS_KWIC = PRIVATE / "validation_terms_kwic.csv"
RESULTS = OUTPUTS / "validation_results.json"
SCRIPT_COLUMNS = ["markup_correct (yes/no)", "final_script_or_transcript (final/transcript/unsure)",
                  "imdb_id_mapping_correct (yes/no)", "character_names_extracted_correctly (yes/partly/no)",
                  "checked_by", "notes"]
TERM_COLUMNS = ["category (name/franchise/artefact/language)", "checked_by", "second_checker_category", "notes"]
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def run() -> dict:
    c = cfg()
    rng = np.random.default_rng(c["seed"])
    sc = s01_groups.load()["scripts"].sort_values("tconst").reset_index(drop=True)
    rep = s02_layers.load()["report"].set_index("tconst")
    lay = s02_layers.load_layers().set_index("tconst")
    pick = sc.iloc[sorted(rng.choice(len(sc), c["validation"]["scripts"], replace=False))]
    sheet = pick[["tconst", "movie_name", "budget_usd", "gross_usd"]].copy()
    r = rep.loc[pick.tconst]
    for k in ("n_scenes", "n_cues", "dialogue_words", "action_words", "full_words"):
        sheet[k] = r[k].to_numpy()
    sheet["parse_error"] = r.parse_error.fillna("").to_numpy()
    for col in SCRIPT_COLUMNS:
        sheet[col] = ""
    if not SCRIPTS_SHEET.exists() or _empty(pd.read_csv(SCRIPTS_SHEET, dtype=str), SCRIPT_COLUMNS[:4]):
        sheet.to_csv(SCRIPTS_SHEET, index=False)
    prev = [{"tconst": t, "first 15 dialogue words": " ".join(WORD.findall(lay.loc[t, "dialogue"])[:15]),
             "first 15 action words": " ".join(WORD.findall(lay.loc[t, "action"])[:15]),
             "first heading": lay.loc[t, "headings"].split("\n")[0][:80]} for t in pick.tconst]
    pd.DataFrame(prev).to_csv(SCRIPTS_PREVIEW, index=False)

    terms = s05_scatter.load()["primary"].sort_values("term").reset_index(drop=True)
    tp = terms.iloc[sorted(rng.choice(len(terms), c["validation"]["terms"], replace=False))]
    tsheet = tp[["term", "score", "doc_freq"]].copy()
    for col in TERM_COLUMNS:
        tsheet[col] = ""
    if not TERMS_SHEET.exists() or _empty(pd.read_csv(TERMS_SHEET, dtype=str), TERM_COLUMNS[:1]):
        tsheet.to_csv(TERMS_SHEET, index=False)
    kw = []
    for t in tp.term:
        found = 0
        for tid in lay.index[rng.permutation(len(lay))]:
            w = s08_kwic.windows(lay.loc[tid, "dialogue"], t, c["kwic"]["window_words"], 1)
            if w:
                kw.append({"term": t, "tconst": tid, "window": w[0]})
                found += 1
            if found >= 3:
                break
    pd.DataFrame(kw).to_csv(TERMS_KWIC, index=False)
    log_run("s11_validation", [SCRIPTS_SHEET.name, TERMS_SHEET.name], {"validation": c["validation"], "seed": c["seed"]})
    score()
    return load()


def _empty(df, cols) -> bool:
    return all(df[c].fillna("").str.strip().eq("").all() for c in cols if c in df)


def score() -> dict:
    """Agreement and precision per category from the filled sheets; 'not yet validated' while empty."""
    out = {"status": "NOT YET VALIDATED"}
    s = pd.read_csv(SCRIPTS_SHEET, dtype=str).fillna("")
    t = pd.read_csv(TERMS_SHEET, dtype=str).fillna("")
    if not _empty(s, SCRIPT_COLUMNS[:4]):
        out["scripts_checked"] = int((s[SCRIPT_COLUMNS[0]].str.strip() != "").sum())
        for col in SCRIPT_COLUMNS[:4]:
            v = s[col].str.strip().str.lower()
            v = v[v != ""]
            out[f"scripts: {col}"] = v.value_counts(normalize=True).round(3).to_dict()
    if not _empty(t, TERM_COLUMNS[:1]):
        a, b = t[TERM_COLUMNS[0]].str.strip().str.lower(), t["second_checker_category"].str.strip().str.lower()
        filled = a != ""
        out["terms_checked"] = int(filled.sum())
        out["terms: share real language"] = float((a[filled] == "language").mean())
        out["terms: category shares"] = a[filled].value_counts(normalize=True).round(3).to_dict()
        both = filled & (b != "")
        if both.sum():
            out["terms: agreement between checkers"] = float((a[both] == b[both]).mean())
            out["terms: double-checked"] = int(both.sum())
    if len(out) > 1:
        out["status"] = "validated (partly)" if out.get("terms_checked", 0) < len(t) or \
            out.get("scripts_checked", 0) < len(s) else "validated"
    RESULTS.write_text(json.dumps(out, indent=1) + "\n")
    return out


def load() -> dict:
    return {"scripts": pd.read_csv(SCRIPTS_SHEET, dtype={"tconst": str}), "terms": pd.read_csv(TERMS_SHEET),
            "results": json.loads(RESULTS.read_text())}


def tables(res: dict) -> dict:
    r = res["results"]
    return {"scripts": res["scripts"], "terms": res["terms"],
            "results": pd.DataFrame([{"measure": k, "value": json.dumps(v) if isinstance(v, dict) else v}
                                     for k, v in r.items()])}


def figures(res: dict) -> dict:
    return {}


def reading(res: dict) -> str:
    r = res["results"]
    if r["status"] == "NOT YET VALIDATED":
        return ("**NOT YET VALIDATED.** The two sheets are ready but nobody has filled them in. Until they are, "
                "nothing in these notebooks has been checked by a person against the scripts.")
    parts = [f"Status: {r['status']}."]
    if "terms_checked" in r:
        parts.append(f"{r['terms_checked']} terms checked; {r['terms: share real language']:.0%} judged real language.")
    if "terms: agreement between checkers" in r:
        parts.append(f"Two checkers agreed on {r['terms: agreement between checkers']:.0%} of "
                     f"{r['terms: double-checked']} double-checked terms.")
    if "scripts_checked" in r:
        parts.append(f"{r['scripts_checked']} scripts checked.")
    return " ".join(parts)


SECTION = [
    Md("""## Step 11 · Human validation (prepared here, filled in by the team)

**What we do.** We prepare two sheets with fixed random samples (seed in `config.yaml`):

- **(a) 25 scripts**: IMDb ID, title, budget, gross, what the parser found, and empty columns: is the markup
  right? Is it a final script or a transcript? Is the IMDb ID mapped to the right film? Were the character
  names extracted correctly? A private preview with the first 15 words of each layer sits next to it.
- **(b) 30 Scattertext terms**: the term, its score and document frequency, and an empty column: is it a
  name, a franchise word, a formatting artefact, or real language? A second checker column measures
  agreement. Short KWIC lines sit next to it in the private folder.

**Why.** Everything before this step is a machine's reading. The course asks for validation on our own
sample: the numbers mean what we say only if a person, reading the text, agrees.

**What it measures.** Once filled: the share of scripts that parse correctly and are final scripts, the share
of terms that are real language, and agreement between two checkers.

**What it misses.** 25 scripts and 30 terms are small samples; they catch systematic problems, not rare ones.

**How to fill it in.** Open `analysis/scripts_text/outputs/validation_scripts.csv` and
`validation_terms.csv`, read the matching files in `analysis/scripts_text/private/`, fill the empty columns,
then rerun this notebook (or `python -m analysis.scripts_text.s11_validation score`)."""),
    Run("s11_validation"),
    Table("s11_validation", "results", "Validation status, computed from the sheets."),
    Table("s11_validation", "scripts", "Sheet (a): 25 scripts for the team to check. Empty columns are for people."),
    Table("s11_validation", "terms", "Sheet (b): 30 terms for the team to classify."),
    Reading("s11_validation"),
]


if __name__ == "__main__":
    if sys.argv[1:] == ["score"]:
        print(json.dumps(score(), indent=1))
