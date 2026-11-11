"""Step 3: remove what isn't language (headings, technical directions, page artefacts, character
names), document every removal, and build the token streams every later step uses."""
from __future__ import annotations

import html
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import s01_groups, s02_layers
from .config import cfg, log_run
from .paths import OUTPUTS, PRIVATE
from .report import ACCENT, HIGH, LOW, NEUTRAL, Fig, Md, Private, Reading, Run, Table, style

TOKENS = PRIVATE / "tokens.parquet"        # one row per script x layer, token lists: never committed
NAMES = PRIVATE / "names.parquet"          # per-script name lists (labels and NER)
STAGES = OUTPUTS / "s03_stages.csv"        # tokens / types / hapax after each cleaning stage
TOP = OUTPUTS / "s03_top_terms.csv"        # top-30 terms before and after
RANKFREQ = OUTPUTS / "s03_rank_frequency.csv"
NAME_REPORT = OUTPUTS / "s03_names.csv"    # per-script counts only
SUMMARY = OUTPUTS / "s03_summary.json"
DICT = Path("/usr/share/dict/words")

TOKEN = re.compile(cfg()["text"]["token_regex"])
SPLIT_CONTRACTION = re.compile(r"(\w)\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", re.I)
TECHNICAL = re.compile(
    r"\b(?:CUT TO|SMASH CUT(?: TO)?|MATCH CUT(?: TO)?|JUMP CUT(?: TO)?|FADE IN|FADE OUT|FADE TO(?: BLACK| WHITE)?|"
    r"DISSOLVE(?: TO)?|INTERCUT(?: WITH)?|BACK TO(?: SCENE| PRESENT)?|CONTINUED|CONT'D|CONT'D|CONTD|"
    r"ANGLE ON|CLOSE ON|CLOSE UP|CLOSE-UP|WIDE SHOT|INSERT|POV|SUPER|TITLE CARD|THE END|MORE|"
    r"V\.O\.|O\.S\.|O\.C\.|OS|VO|OC)(?!\w):?")
ARTEFACT_LINE = re.compile(
    r"^\s*(?:\d+[A-Z]?\.?|page \d+|\*+|\(?rev(?:\.|ised|isions?)\s*(?:pages?\s*)?\(?\d+[/.-]\d+.*|"
    r"(?:written|screenplay|story|teleplay|directed|adaptation) by\b.*|"
    r"based on (?:the|a|an) (?:novel|book|story|play|short story|comic)\b.*|"
    r"(?:final|shooting|first|second|revised) (?:draft|script)\b.*|"
    r"(?:white|blue|pink|yellow|green|goldenrod|buff|salmon|cherry|tan) (?:revisions?|pages?)\b.*)\s*$", re.I)
NOT_A_NAME = re.compile(r"\b(THE END|CUT TO|FADE|DISSOLVE|SCRIPT|DRAFT|REVISION|CONTINUED|SMASH CUT)\b|^(INT|EXT)\b",
                        re.I)
TITLE_WORDS = {"mr", "mrs", "ms", "miss", "dr", "sir", "lady", "lord", "capt", "sgt", "lt", "col", "gen", "det",
               "the", "a", "an", "and", "of", "o", "s", "v", "c", "jr", "sr", "ii", "iii"}


# ---------------------------------------------------------------- cleaning

def dictionary() -> set[str]:
    return {w.strip() for w in DICT.read_text().splitlines() if w and w[0].islower()}


def label_names(cues: list[str]) -> set[str]:
    """Name tokens from a script's own speaker cues: strip extensions, drop headers and titles."""
    out = set()
    for raw in cues:
        lab = html.unescape(raw)
        if NOT_A_NAME.search(lab):
            continue
        lab = re.sub(r"\([^)]*\)", " ", lab)
        lab = re.sub(r"\bCONT'?D\b|#\d+|\d+", " ", lab, flags=re.I)
        for tok in TOKEN.findall(lab):
            t = tok.lower().removesuffix("'s")
            if len(t) >= 2 and t not in TITLE_WORDS:
                out.add(t)
    return out


def clean_lines(text: str, counts: Counter, headings: set[str] | None = None) -> str:
    """Contractions, headings, artefact lines and technical directions, each removal counted."""
    joined = SPLIT_CONTRACTION.sub(r"\1\2", text)
    counts["contraction_joins"] += len(SPLIT_CONTRACTION.findall(text))
    keep = []
    for line in joined.split("\n"):
        if headings is not None and line.strip() in headings:
            counts["heading_tokens"] += len(TOKEN.findall(line))
            continue
        if ARTEFACT_LINE.match(line):
            counts["artefact_tokens"] += len(TOKEN.findall(line))
            continue
        n = len(TOKEN.findall(line))
        line = TECHNICAL.sub(" ", line)
        counts["technical_tokens"] += n - len(TOKEN.findall(line))
        keep.append(line)
    return "\n".join(keep)


def name_mask(text: str, names: set[str], dict_words: set[str]) -> tuple[list[str], list[bool]]:
    """Tokens (lowercase) and whether each is a name to remove. A name that is also a dictionary
    word ('will', 'grace') is removed only where it is capitalised and not starting a sentence."""
    toks, is_name = [], []
    for m in TOKEN.finditer(text):
        t = m.group(0)
        low = t.lower()
        toks.append(low)
        base = low.removesuffix("'s")
        if base not in names:
            is_name.append(False)
            continue
        if base not in dict_words:
            is_name.append(True)
            continue
        before = text[max(0, m.start() - 3):m.start()].strip()
        sentence_start = before == "" or before[-1] in ".!?\n:\"-"
        is_name.append(t[0].isupper() and not sentence_start)
    return toks, is_name


# ---------------------------------------------------------------- the heavy pass

def _spacy_dialogue(texts: dict[str, str]) -> dict[str, dict]:
    """One spaCy pass over every cleaned dialogue layer: lemma per character offset, PERSON tokens."""
    import spacy
    nlp = spacy.load(cfg()["text"]["spacy_model"], disable=["parser"])
    nlp.max_length = 2_000_000
    step = cfg()["text"]["spacy_batch_chars"]
    ids = sorted(texts)
    chunks, owner = [], []
    for tid in ids:
        t, start = texts[tid], 0
        while start < len(t):
            end = t.rfind("\n", start, start + step)
            end = len(t) if end <= start or start + step >= len(t) else end
            chunks.append((t[start:end], start))
            owner.append(tid)
            start = end + 1 if end < len(t) else len(t)
    out = {tid: {"lemma": {}, "person": Counter()} for tid in ids}
    t0 = time.time()
    docs = nlp.pipe((c for c, _ in chunks), n_process=6, batch_size=8)
    for i, (doc, (_, offset), tid) in enumerate(zip(docs, chunks, owner), 1):
        lem = out[tid]["lemma"]
        for tok in doc:
            lem[offset + tok.idx] = tok.lemma_.lower()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                for w in TOKEN.findall(ent.text):
                    out[tid]["person"][w.lower().removesuffix("'s")] += 1
        if i % max(1, len(chunks) // 20) == 0 or i == len(chunks):
            el = time.time() - t0
            print(f"spaCy: {i}/{len(chunks)} chunks ({i / len(chunks):.0%}), {el / 60:.1f} min, "
                  f"about {el / i * (len(chunks) - i) / 60:.1f} min left", flush=True)
    return out


def prepare(force: bool = False) -> None:
    """Clean every layer, find names, run spaCy on dialogue, and cache token streams in private/."""
    if TOKENS.exists() and not force:
        return
    lay = s02_layers.load_layers()
    dict_words = dictionary()
    from spacy.lang.en.stop_words import STOP_WORDS as stop
    cleaned, counts_by = {}, {}
    for r in lay.itertuples():
        hd = {h.strip() for h in r.headings.split("\n") if h.strip()}
        c = Counter()
        cleaned[r.tconst] = {"dialogue": clean_lines(r.dialogue, c),
                             "action": clean_lines(r.action, c),
                             "full": clean_lines(r.full, c, headings=hd)}
        counts_by[r.tconst] = c
    sp = _spacy_dialogue({t: v["dialogue"] for t, v in cleaned.items()})

    rows, name_rows = [], []
    for r in lay.itertuples():
        names = label_names(json.loads(r.cues))
        ner = {w for w, n in sp[r.tconst]["person"].items() if len(w) >= 2 and w not in TITLE_WORDS}
        for layer in ("dialogue", "action", "full"):
            text = cleaned[r.tconst][layer]
            toks, is_name = name_mask(text, names, dict_words)
            _, is_name_ner = name_mask(text, names | ner, dict_words)
            lemmas = None
            if layer == "dialogue":
                lem = sp[r.tconst]["lemma"]
                lemmas = [lem.get(m.start(), m.group(0).lower()) for m in TOKEN.finditer(text)]
            bits = lambda xs: "".join("1" if x else "0" for x in xs)
            rows.append({"tconst": r.tconst, "layer": layer, "tokens": " ".join(toks), "is_name": bits(is_name),
                         "is_name_ner": bits(is_name_ner), "lemmas": " ".join(lemmas) if lemmas else None,
                         "is_stop": bits(t in stop for t in toks)})
        n_dict = sum(1 for n in names if n in dict_words)
        name_rows.append({"tconst": r.tconst, "label_names": sorted(names), "ner_names": sorted(ner),
                          **{k: int(v) for k, v in counts_by[r.tconst].items()}, "n_label_names": len(names),
                          "n_label_names_dictionary": n_dict, "n_ner_names": len(ner),
                          "n_ner_not_in_labels": len(ner - names)})
    pd.DataFrame(rows).to_parquet(TOKENS, index=False)
    pd.DataFrame(name_rows).to_parquet(NAMES, index=False)


# ---------------------------------------------------------------- token streams for later steps

def load_tokens() -> pd.DataFrame:
    return pd.read_parquet(TOKENS)


def _lemma(lemma: str, token: str) -> str:
    m = TOKEN.search(lemma)
    return m.group(0).lower() if m else token


def stream(row, names: str = "labels", stopwords: str = "kept", lemmatize: str = "off") -> list[str]:
    """One script-layer's tokens under a preprocessing variant. names: 'kept', 'labels' or 'labels+ner'."""
    use_lemma = lemmatize == "on" and isinstance(row["lemmas"], str)
    toks = row["tokens"].split() if isinstance(row["tokens"], str) else []
    if use_lemma:
        # spaCy keeps the period in abbreviations ("mr.", "l.a."): normalise each lemma with our tokenizer
        # so it lines up with the unlemmatised word.
        toks = [_lemma(l, t) for l, t in zip(row["lemmas"].split(), toks)]
    mask = {"kept": None, "labels": row["is_name"], "labels+ner": row["is_name_ner"]}[names]
    stops = row["is_stop"]
    min_len = cfg()["text"]["min_token_length"]
    out = []
    for i, t in enumerate(toks):
        if mask is not None and mask[i] == "1":
            continue
        if stopwords == "removed" and stops[i] == "1":
            continue
        if len(t) < min_len:
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------- statistics

def _tth(counter: Counter) -> dict:
    return {"tokens": int(sum(counter.values())), "types": int(len(counter)),
            "hapax": int(sum(1 for v in counter.values() if v == 1))}


def run() -> dict:
    prepare()
    lay = s02_layers.load_layers().set_index("tconst")
    tok = load_tokens()
    names = pd.read_parquet(NAMES)
    d = tok[tok.layer == "dialogue"].set_index("tconst")
    stages = []
    # Stage 0: raw dialogue as extracted, tokenised the same way.
    raw = Counter(w.lower() for t in lay.dialogue for w in TOKEN.findall(t))
    stages.append({"stage": "0 raw dialogue (Step 2)", **_tth(raw)})
    cleaned = Counter(w for ws in d.tokens.fillna("") for w in ws.split())
    stages.append({"stage": "1 contractions joined, technical directions and artefact lines removed", **_tth(cleaned)})
    variants = {"2 character names removed (labels)": dict(names="labels"),
                "2b names removed (labels + NER), check only": dict(names="labels+ner"),
                "3a + stopwords removed (variant)": dict(names="labels", stopwords="removed"),
                "3b + lemmatised (variant)": dict(names="labels", lemmatize="on")}
    final = {}
    for label, kw in variants.items():
        cnt = Counter(w for _, r in d.iterrows() for w in stream(r, **kw))
        final[label] = cnt
        stages.append({"stage": label, **_tth(cnt)})
    st = pd.DataFrame(stages)
    st["min_length_rule"] = f"tokens shorter than {cfg()['text']['min_token_length']} letters dropped from stage 2 on"
    st.to_csv(STAGES, index=False)

    primary = final["2 character names removed (labels)"]
    top = pd.DataFrame({"rank": range(1, 31),
                        "before (raw)": [w for w, _ in raw.most_common(30)],
                        "before count": [c for _, c in raw.most_common(30)],
                        "after (primary)": [w for w, _ in primary.most_common(30)],
                        "after count": [c for _, c in primary.most_common(30)],
                        "after, stopwords removed": [w for w, _ in final["3a + stopwords removed (variant)"].most_common(30)]})
    top.to_csv(TOP, index=False)
    rf = []
    for label, cnt in (("raw", raw), ("primary", primary), ("stopwords removed", final["3a + stopwords removed (variant)"])):
        freq = np.array(sorted(cnt.values(), reverse=True))
        ranks = np.unique(np.round(np.logspace(0, np.log10(len(freq)), 200)).astype(int))
        rf += [{"version": label, "rank": int(k), "frequency": int(freq[k - 1])} for k in ranks]
    pd.DataFrame(rf).to_csv(RANKFREQ, index=False)

    nm = names.drop(columns=["label_names", "ner_names"])
    removed = d.is_name.str.count("1")
    removed_ner = d.is_name_ner.str.count("1")
    nm = nm.set_index("tconst").assign(dialogue_name_tokens_removed=removed,
                                       dialogue_name_tokens_removed_with_ner=removed_ner).reset_index()
    nm.to_csv(NAME_REPORT, index=False)
    affected = int((nm.contraction_joins >= 20).sum())
    summary = {
        "scripts": int(len(nm)),
        "scripts_pretokenised": affected, "pretokenised_rule": "20 or more split contractions joined",
        "contraction_joins": int(nm.contraction_joins.sum()),
        "technical_tokens_removed": int(nm.technical_tokens.sum()),
        "artefact_tokens_removed": int(nm.artefact_tokens.sum()),
        "heading_tokens_removed_full": int(nm.heading_tokens.sum()),
        "median_label_names": float(nm.n_label_names.median()),
        "label_names_total": int(nm.n_label_names.sum()),
        "label_names_dictionary_words": int(nm.n_label_names_dictionary.sum()),
        "median_ner_names": float(nm.n_ner_names.median()),
        "median_ner_not_in_labels": float(nm.n_ner_not_in_labels.median()),
        "dialogue_name_tokens_removed": int(nm.dialogue_name_tokens_removed.sum()),
        "dialogue_name_tokens_removed_with_ner": int(nm.dialogue_name_tokens_removed_with_ner.sum()),
        "dialogue_tokens_primary": int(sum(primary.values())),
    }
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    log_run("s03_clean", [STAGES.name, TOP.name, RANKFREQ.name, NAME_REPORT.name, SUMMARY.name],
            {"text": cfg()["text"]})
    return load()


def load() -> dict:
    return {"stages": pd.read_csv(STAGES), "top": pd.read_csv(TOP), "rankfreq": pd.read_csv(RANKFREQ),
            "names": pd.read_csv(NAME_REPORT, dtype={"tconst": str}), "summary": json.loads(SUMMARY.read_text())}


def tables(res: dict) -> dict:
    choices = pd.DataFrame([
        {"choice": "tokenizer", "setting": cfg()["text"]["token_regex"], "why": "same as the reviews analysis; keeps don't, it's as one word"},
        {"choice": "lowercase", "setting": "yes", "why": "'The' and 'the' are one word for our question"},
        {"choice": "split contractions", "setting": "joined ('do n't' → 'don't')", "why": "some MovieSum scripts are pre-tokenised; without this 'n't' becomes a word"},
        {"choice": "scene headings", "setting": "removed (full layer)", "why": "INT., EXT., DAY, NIGHT describe format, not language"},
        {"choice": "technical directions", "setting": "removed (CUT TO, FADE IN, V.O., O.S., CONT'D …)", "why": "formatting, and some writers use far more of it"},
        {"choice": "page and revision artefacts", "setting": "lines removed (page numbers, 'Revised …', 'Written by …')", "why": "not part of the story"},
        {"choice": "character names", "setting": "removed (from each script's own speaker cues)", "why": "names are unique to a film and would dominate the comparison"},
        {"choice": "names that are words", "setting": "removed only when capitalised mid-sentence", "why": "keeps 'will' the verb while removing 'Will' the character"},
        {"choice": "minimum token length", "setting": f"{cfg()['text']['min_token_length']} letters", "why": "single letters are mostly initials and artefacts"},
        {"choice": "stopwords", "setting": "kept (primary); removed (variant, spaCy list)", "why": "function words carry style; removing them is a modelling choice"},
        {"choice": "lemmatisation", "setting": "off (primary); on (variant, spaCy, dialogue only)", "why": "lemmas merge tense and number, which can be the signal"},
    ])
    return {"choices": choices, "stages": res["stages"], "top": res["top"]}


def private(res: dict) -> dict:
    """20 scripts at random (fixed seed) with the names found: notebook only."""
    nm = pd.read_parquet(NAMES)
    rng = np.random.default_rng(cfg()["seed"])
    pick = nm.iloc[sorted(rng.choice(len(nm), 20, replace=False))]
    titles = s01_groups.load()["scripts"].set_index("tconst").movie_name
    return {"names_sample": pd.DataFrame({
        "title": pick.tconst.map(titles).values,
        "label names (first 12)": [", ".join(list(v)[:12]) for v in pick.label_names],
        "NER names not in labels (first 8)": [", ".join([w for w in v if w not in set(l)][:8])
                                               for v, l in zip(pick.ner_names, pick.label_names)]})}


def figures(res: dict) -> dict:
    rf = res["rankfreq"]
    fig = go.Figure()
    for version, colour, dash in (("raw", NEUTRAL, "dot"), ("primary", HIGH, "solid"), ("stopwords removed", LOW, "dash")):
        v = rf[rf.version == version]
        fig.add_trace(go.Scatter(x=v["rank"], y=v.frequency, mode="lines", name=version, line=dict(color=colour, dash=dash, width=2.5)))
    fig.update_xaxes(type="log", title_text="rank of the word (1 = most frequent), log scale")
    fig.update_yaxes(type="log", title_text="times the word occurs in all dialogue, log scale")
    style(fig, "Rank-frequency of dialogue words, before and after cleaning", height=420)

    st = res["stages"]
    bar = go.Figure()
    for col, colour, pat in (("tokens", HIGH, ""), ("types", ACCENT, "/"), ("hapax", LOW, "x")):
        bar.add_trace(go.Bar(y=st.stage, x=st[col], name=col, orientation="h", marker_color=colour, marker_pattern_shape=pat))
    bar.update_layout(barmode="group", xaxis_type="log", xaxis_title="count, log scale", yaxis_autorange="reversed")
    style(bar, "Tokens, types and hapax after each cleaning stage (dialogue)", height=460)
    return {"rankfreq": fig, "stages": bar}


def reading(res: dict) -> str:
    s, st = res["summary"], res["stages"].set_index("stage")
    raw, prim = st.iloc[0], st.loc["2 character names removed (labels)"]
    return (f"Cleaning joined {s['contraction_joins']:,} split contractions; {s['scripts_pretokenised']} scripts "
            f"are pre-tokenised ({s['pretokenised_rule']}). It removed {s['technical_tokens_removed']:,} tokens of "
            f"technical directions and {s['artefact_tokens_removed']:,} tokens on page or revision lines across all "
            f"layers, and {s['heading_tokens_removed_full']:,} heading tokens from the full layer. Speaker cues give a "
            f"median of {s['median_label_names']:.0f} name tokens per script; {s['label_names_dictionary_words']:,} of "
            f"the {s['label_names_total']:,} are also ordinary English words, which is why those are removed only when "
            f"capitalised mid-sentence. In the dialogue, {s['dialogue_name_tokens_removed']:,} name tokens go "
            f"({s['dialogue_name_tokens_removed']/raw.tokens:.1%} of all dialogue tokens). spaCy's NER finds a median "
            f"of {s['median_ner_names']:.0f} person-name tokens per script, {s['median_ner_not_in_labels']:.0f} of them "
            f"not among the speaker cues (people who are talked about but never speak, and NER errors); adding them "
            f"would remove {s['dialogue_name_tokens_removed_with_ner'] - s['dialogue_name_tokens_removed']:,} more "
            f"tokens. The primary dialogue corpus has {int(prim.tokens):,} tokens, {int(prim.types):,} types and "
            f"{int(prim.hapax):,} hapax types (tokens ≥ types ≥ hapax holds at every stage).")


SECTION = [
    Md("""## Step 3 · Cleaning what isn't language

**What we do.** We remove what a reader wouldn't count as the script's language, and count every removal:

- **split contractions** joined back ("do n't" → "don't"), because some scripts in MovieSum are pre-tokenised;
- **scene headings** (INT., EXT., DAY, NIGHT) and **technical directions** (CUT TO, FADE IN, V.O., O.S., CONT'D);
- **page and revision artefacts** (page numbers, "Revised pages", "Written by …");
- **character names**, taken from each script's own speaker cues.

**Why names.** A character's name is unique to one film, so in a two-group comparison it is always
"distinctive" and tells us nothing about success. In the course's Marvel text, names dominated every
word list for this reason.

**Extraction is not resolution.** The speaker cues are raw strings ("JULES (V.O.)", "COP #2", "OLD MAN"). We
strip extensions and titles and split them into tokens, but we don't resolve nicknames ("Mike" vs "Michael").
Some name tokens are ordinary words (Will, Grace, Hope, Frank): those are removed only where they are
capitalised in the middle of a sentence. As a second source of names we run spaCy's named-entity recogniser
(NER) on the dialogue and report how much it would add, without using it in the primary analysis.

**Preprocessing is part of the model.** Every choice below changes which words can appear distinctive later.
We keep stopwords (function words like "the", "you", "we" carry style) and don't lemmatise (lemmas merge
"killed" and "kills", which may be the signal), and run both the other way as variants.

**What it measures.** The vocabulary that is left: tokens (every occurrence), types (distinct words) and hapax
types (words occurring once). Tokens ≥ types ≥ hapax always.

**What it misses.** Our tokenizer splits on anything that isn't a letter, so "ice-cream" becomes two words
and numbers disappear. The name list misses characters who never speak and keeps nicknames unresolved."""),
    Run("s03_clean"),
    Table("s03_clean", "choices", "Every preprocessing choice, its setting and why."),
    Table("s03_clean", "stages", "Tokens, types and hapax types of all dialogue after each stage. Stage 2 is "
          "the primary corpus; 2b, 3a and 3b are variants."),
    Fig("s03_clean", "stages", "The same counts as bars on a log axis. Names remove many types but few tokens; "
        "stopwords remove few types but many tokens."),
    Table("s03_clean", "top", "The 30 most frequent dialogue words before cleaning, after cleaning (primary) "
          "and with stopwords removed. Before cleaning, names and formatting can appear; after, the list is "
          "ordinary spoken English."),
    Fig("s03_clean", "rankfreq", "Rank against frequency on log-log axes for the raw, primary and "
        "stopwords-removed dialogue. A roughly straight line is suggestive of a Zipf-like heavy tail: a few words "
        "are extremely common and most are rare. It is not proof of Zipf's law; testing that would need a "
        "fitted model against alternatives."),
    Private("s03_clean", "names_sample", "20 scripts at random (fixed seed) with the name tokens found in their "
            "speaker cues and the extra ones NER suggests. Check: are these names, and is anything missing? "
            "Notebook only, output cleared before commit."),
    Reading("s03_clean"),
]
