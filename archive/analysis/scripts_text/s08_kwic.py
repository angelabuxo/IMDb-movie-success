"""Step 8: go back to the text. Short keyword-in-context windows for the top terms."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import s01_groups, s02_layers, s05_scatter, s06_null
from .config import cfg, log_run
from .paths import OUTPUTS, PRIVATE
from .report import Md, Private, Reading, Run, Table

TERMS = OUTPUTS / "s08_terms.csv"            # counts and an empty human column: committed
KWIC = PRIVATE / "s08_kwic.csv"              # the windows themselves: never committed
SUMMARY = OUTPUTS / "s08_summary.json"
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
CATEGORIES = "name / franchise word / formatting artefact / topic word / style word / other"


NAME_IN_SCRIPTS = 3   # a term is flagged as a name only if it is a speaker cue in at least this many scripts


def hint(n_name_scripts: int, top_share: float) -> str:
    """A machine hint, not the human category: name, one-film word, or nothing."""
    parts = []
    if n_name_scripts >= NAME_IN_SCRIPTS:
        parts.append(f"speaker name in {n_name_scripts} scripts' cues")
    if top_share > 0.5:
        parts.append("one script carries over half its uses")
    return "; ".join(parts)


def windows(text: str, term: str, k: int, limit: int):
    """Up to `limit` windows of k words either side of the term (2k + 1 <= 15 words)."""
    words = WORD.findall(text)
    low = [w.lower() for w in words]
    out = []
    for i, w in enumerate(low):
        if w == term:
            out.append(" ".join(words[max(0, i - k):i]) + " [" + words[i] + "] " + " ".join(words[i + 1:i + 1 + k]))
            if len(out) >= limit:
                break
    return out


def run() -> dict:
    c = cfg()
    k, per, n = c["kwic"]["window_words"], c["kwic"]["per_term"], c["kwic"]["top_terms_per_side"]
    assert 2 * k + 1 <= 15
    prim = s05_scatter.load()["primary"]
    X, vocab, ids, lengths, y, _ = s06_null.primary_inputs()
    col = {t: i for i, t in enumerate(vocab)}
    titles = s01_groups.load()["scripts"].set_index("tconst").movie_name
    lay = s02_layers.load_layers().set_index("tconst")
    from . import s03_clean
    names = pd.read_parquet(s03_clean.NAMES)
    name_anywhere = {}
    for v in names.label_names:
        for w in v:
            name_anywhere[w] = name_anywhere.get(w, 0) + 1

    rows, kw = [], []
    rng = np.random.default_rng(c["seed"])
    for side, terms in (("successful", prim.head(n)), ("unsuccessful", prim.tail(n).iloc[::-1])):
        for r in terms.itertuples():
            j = col[r.term]
            colj = X[:, j].toarray().ravel()
            total = int(colj.sum())
            top_doc = int(np.argmax(colj))
            docs = np.where(colj > 0)[0]
            rows.append({"side": side, "term": r.term, "score": round(r.score, 3),
                         "scripts_containing": int(len(docs)),
                         "scripts_successful": int((colj[y == 1] > 0).sum()),
                         "scripts_unsuccessful": int((colj[y == 0] > 0).sum()), "total_uses": total,
                         "share_from_top_script": round(colj[top_doc] / total, 3) if total else None,
                         "top_script": titles.get(ids[top_doc], ids[top_doc]),
                         "scripts_where_it_is_a_character_name": name_anywhere.get(r.term, 0),
                         "machine_hint": hint(name_anywhere.get(r.term, 0), colj[top_doc] / total if total else 0),
                         "human_category": "", "human_notes": ""})
            for d in rng.choice(docs, min(per, len(docs)), replace=False):
                for wdw in windows(lay.loc[ids[d], "dialogue"], r.term, k, 1):
                    kw.append({"side": side, "term": r.term, "title": titles.get(ids[d], ids[d]), "window": wdw})
    t = pd.DataFrame(rows)
    t.to_csv(TERMS, index=False)
    pd.DataFrame(kw).to_csv(KWIC, index=False)
    summary = {"terms": int(len(t)), "windows": int(len(kw)), "window_words_max": 2 * k + 1,
               "terms_that_are_names_somewhere": int((t.scripts_where_it_is_a_character_name >= NAME_IN_SCRIPTS).sum()),
               "name_rule_scripts": NAME_IN_SCRIPTS,
               "terms_mostly_one_script": int((t.share_from_top_script > 0.5).sum()),
               "human_categories_filled": int((t.human_category.fillna("") != "").sum())}
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s08_kwic", [TERMS.name, SUMMARY.name], {"kwic": c["kwic"]})
    return load()


def load() -> dict:
    return {"terms": pd.read_csv(TERMS), "summary": json.loads(SUMMARY.read_text())}


def tables(res: dict) -> dict:
    return {"terms": res["terms"]}


def private(res: dict) -> dict:
    return {"kwic": pd.read_csv(KWIC)}


def figures(res: dict) -> dict:
    return {}


def takeaway(res: dict) -> str:
    s = res["summary"]
    status = ("Nobody has classified them by hand yet." if s["human_categories_filled"] == 0 else
              f"{s['human_categories_filled']} of {s['terms']} have a human category so far.")
    return (f"Of the {s['terms']} most extreme words, {s['terms_that_are_names_somewhere']} are a speaker name in at "
            f"least {s['name_rule_scripts']} scripts, and {s['terms_mostly_one_script']} get more than half their uses "
            f"from a single script. {status}")


def reading(res: dict) -> str:
    s = res["summary"]
    status = ("NOT YET VALIDATED. The human category column is empty." if s["human_categories_filled"] == 0 else
              f"{s['human_categories_filled']} of {s['terms']} terms have a human category.")
    return (f"Of the {s['terms']} most extreme terms ({s['terms'] // 2} per side), "
            f"{s['terms_that_are_names_somewhere']} are a speaker name in the cues of at least "
            f"{s['name_rule_scripts']} scripts. We ask for {s['name_rule_scripts']} because MovieSum's cues also hold "
            f"mis-tagged headings, so one or two scripts aren't enough. Name removal works script by script, so it "
            f"can miss these names when a character is talked about in another film, or when a name is also a word. "
            f"{s['terms_mostly_one_script']} terms get more than half their uses from a single script. "
            f"{s['windows']} windows of at most {s['window_words_max']} words are written to the private folder for "
            f"reading. {status} If the table and the text disagree, the text wins.")


SECTION = [
    Md("""## Step 8 · Go back to the text

This step asks what the most distinctive words really are once you read them in their scripts.

### What we do

For the 15 most extreme terms on each side of the primary run, we count how many scripts use the term, what
share of its uses comes from the single script that uses it most, and whether it's a character name in some
other script. We then pull a handful of keyword-in-context (KWIC) windows per term, from different scripts. A
KWIC window is the term with a few words on either side, at most 15 words in all.

### Why

A word list is a summary, and the text is the evidence. A "distinctive" word can turn out to be one film's
catchphrase, a name, a formatting leftover or a genre topic, and only reading tells which. **This
classification is a human decision.** The table has an empty column for it, and nothing here fills it in.

### What it measures

How concentrated each term is, and what its uses look like.

### What it misses

Five windows per term is a glimpse, not a reading of every use."""),
    Run("s08_kwic"),
    Table("s08_kwic", "terms", "The top 15 terms per side: how many scripts use each one, the share of its uses "
          "that comes from the single heaviest script, whether it's a character name elsewhere, a machine hint, "
          f"and empty columns for the human category ({CATEGORIES}) and for notes."),
    Private("s08_kwic", "kwic", "Keyword-in-context windows of at most 15 words. They're in the notebook only, "
            "with the output cleared before commit, and the full file is in the private folder."),
    Reading("s08_kwic"),
]
