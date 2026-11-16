"""Cleaning rules for the three corpora. Pure functions, tested in tests/test_embeddings_clean.py.

What each rule is for:
- markup, entities, URLs and tokeniser spacing go, because they are how the text was stored, not what it says;
- review sentences that talk about money go, because a review is written after release and "it flopped" is
  the outcome we correlate with, leaking into the predictor;
- the film's own title and the names of people go, because a language model has read about famous films and
  stars: with names left in, a vector can encode who made the film (the network half of the project) or which
  film it is, instead of what the text says;
- screenplay scene headings, speaker labels, parentheticals and transitions go, because they are formatting.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter

TAG = re.compile(r"<[^<>]{1,80}>")
URL = re.compile(r"https?://\S+|www\.\S+")
WS = re.compile(r"\s+")
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]*[A-Z0-9])")

# Talk about money or commercial outcome. "gross" alone is left: in reviews it is mostly the adjective.
MONEY = re.compile(
    r"box[\s-]?office|\bgross(?:ed|ing|es)\b|\b(?:worldwide|domestic|total|opening) gross|highest[\s-]grossing"
    r"|\bflop(?:s|ped|ping)?\b|\bbomb(?:ed|ing)\b|blockbuster|\bbudget(?:s|ed)?\b|[$€£]\s?\d|\bdollars?\b"
    r"|\b\d+(?:\.\d+)?\s?(?:million|billion|mil|m|bn)\b|\b(?:million|billion)s?\b|\bprofit(?:s|able|ed)?\b"
    r"|\b(?:made|make|makes|making|lost|lose|loses) (?:a lot of |so much |its |their |some |any |more |big |no )?money\b"
    r"|ticket sales|tickets sold|opening weekend|commercial(?:ly)? (?:success|successful|failure|hit|flop)"
    r"|\b(?:smash|big|huge|biggest|surprise|sleeper|massive|runaway|monster) hit\b|cash[\s-]?(?:cow|grab|in)\b"
    r"|money[\s-]?(?:maker|making|grab)|financial(?:ly)? (?:success|successful|failure|disaster)", re.I)

FUNCTION_WORDS = {"the", "and", "of", "to", "a", "is", "in", "it", "that", "this", "was", "for"}

# Screenplay transitions and page furniture, inside a description or dialogue block.
FURNITURE = re.compile(
    r"\(\s*(?:MORE|CONTINUED|CONT'?D\.?)\s*\)|\b(?:CUT|DISSOLVE|SMASH CUT|MATCH CUT|JUMP CUT|FADE|WIPE) (?:TO|IN|OUT)\b:?"
    r"|\bFADE (?:TO BLACK|UP)\b[.:]?|\bCONTINUED\b:?|\bCONT'D\b|\bTHE END\b|\b(?:INT|EXT)\.(?=\s)|\bO\.S\.|\bV\.O\.")
SCENE = re.compile(r"<scene>(.*?)</scene>", re.S)
BLOCK = re.compile(r"<(character|dialogue|parenthetical|scene_description|stage_direction)>(.*?)</\1>", re.S)
PAREN_NAME = re.compile(r"\s*\((?:[A-Z][\w.'’-]+\s?){1,4}\)")     # "(Tom Hanks)" after a character in a summary
BRACKET = re.compile(r"\s*\[[^\]]{0,40}\]")                       # "[1]", "[citation needed]"


def basic(text: str) -> str:
    """Entities decoded, Unicode normalised, tags and URLs removed, whitespace collapsed."""
    t = html.unescape(html.unescape(text or ""))
    t = unicodedata.normalize("NFKC", t).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = URL.sub(" ", TAG.sub(" ", t))
    return WS.sub(" ", t).strip()


def detokenise(text: str) -> str:
    """MovieSum stores text tokenised ("church . He did n't"). Put it back the way people write it."""
    t = re.sub(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", text, flags=re.I)
    t = re.sub(r"\s+([.,!?;:%)\]])", r"\1", t)
    return re.sub(r"([(\[$])\s+", r"\1", t)


def sentences(text: str) -> list[str]:
    return [s for s in SENT.split(text) if s.strip()]


def n_words(text: str) -> int:
    return len(WORD.findall(text))


def looks_english(text: str, min_ascii: float, min_function: int) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters or sum(c.isascii() for c in letters) / len(letters) < min_ascii:
        return False
    return len(FUNCTION_WORDS & {w.lower() for w in WORD.findall(text[:2000])}) >= min_function


def drop_money_sentences(text: str) -> tuple[str, int, int]:
    """The text without sentences that mention money, the number dropped and the number seen."""
    s = sentences(text)
    kept = [x for x in s if not MONEY.search(x)]
    return " ".join(kept), len(s) - len(kept), len(s)


def title_pattern(titles: list[str]) -> re.Pattern | None:
    """Matches the film's own title. A one-word title is matched only with its capital letter and only
    if it has five letters or more, so "Heat" or "Up" don't erase ordinary words."""
    alts = []
    for raw in titles:
        t = re.sub(r"\s*\(\d{4}\)\s*$", "", raw or "").strip()
        m = re.match(r"^(.*), (The|A|An)$", t)          # "Godfather, The"
        if m:
            t = f"{m.group(2)} {m.group(1)}"
        for form in {t, re.sub(r"^(The|A|An) ", "", t)}:
            if " " in form and len(form) >= 5:
                alts.append("(?i:" + r"\s+".join(map(re.escape, form.split())) + ")")
            elif len(form) >= 5 and form[:1].isupper():
                alts.append(re.escape(form))
    if not alts:
        return None
    return re.compile(r"(?<![A-Za-z])(?:the film |the movie )?(?:" + "|".join(sorted(set(alts), key=len, reverse=True))
                      + r")(?![A-Za-z])")


def people_pattern(names: list[str]) -> re.Pattern | None:
    """Matches full names in any case, and surnames of four letters or more with their capital letter."""
    alts = []
    for name in names:
        parts = [p for p in re.split(r"\s+", (name or "").strip()) if p]
        if len(parts) >= 2:
            alts.append("(?i:" + r"\s+".join(map(re.escape, parts)) + ")")
            if len(parts[-1]) >= 4 and parts[-1][:1].isupper():
                alts.append(re.escape(parts[-1]))
        elif parts and len(parts[0]) >= 4 and parts[0][:1].isupper():
            alts.append(re.escape(parts[0]))
    if not alts:
        return None
    return re.compile(r"(?<![A-Za-z])(?:" + "|".join(sorted(set(alts), key=len, reverse=True)) + r")(?:'s)?(?![A-Za-z])")


def mask(text: str, pattern: re.Pattern | None, placeholder: str) -> tuple[str, int]:
    if pattern is None:
        return text, 0
    return pattern.subn(placeholder, text)


def clean_review(raw: str, title_pat, people_pat, name_ph: str, title_ph: str) -> tuple[str, dict]:
    t, dropped, seen = drop_money_sentences(basic(raw))
    t, n_title = mask(t, title_pat, title_ph)
    t, n_people = mask(t, people_pat, name_ph)
    return WS.sub(" ", t).strip(), {"sentences": seen, "money_sentences": dropped, "titles": n_title, "people": n_people}


# ------------------------------------------------------------------ screenplays

def label_words(raw_label: str) -> list[str]:
    label = re.sub(r"\([^)]*\)", " ", html.unescape(raw_label)).upper()
    label = re.sub(r"\bCONT'?D\b|[^A-Z' -]", " ", label)
    return [w.strip("'-") for w in label.split() if w.strip("'-")]


def script_blocks(script: str) -> tuple[list[str], list[str]]:
    """The description and dialogue blocks in order, and every speaker label. Headings, labels and
    parentheticals are left out of the blocks."""
    blocks, labels = [], []
    for kind, body in BLOCK.findall(script or ""):
        if kind == "character":
            labels.append(body)
        elif kind in ("dialogue", "scene_description"):
            b = WS.sub(" ", FURNITURE.sub(" ", detokenise(basic(body)))).strip(" .:-")
            if WORD.search(b):
                blocks.append(b)
    return blocks, labels


def character_names(text: str, labels: list[str], max_lower: float, min_count: int, min_len: int) -> set[str]:
    """Speaker-label words that are names: written with a capital nearly every time they appear in the
    script. "MAN" or "DOCTOR" also appear in lower case in descriptions, so they stay."""
    cand = {w for lab in labels for w in label_words(lab) if len(w) >= min_len}
    if not cand:
        return set()
    total, lower = Counter(), Counter()
    for w in WORD.findall(text):
        u = w.upper()
        if u in cand:
            total[u] += 1
            lower[u] += w.islower()
    return {w for w in cand if total[w] >= min_count and lower[w] / total[w] <= max_lower}


def clean_script(script: str, max_lower: float, min_count: int, min_len: int, name_ph: str) -> tuple[str, dict]:
    blocks, labels = script_blocks(script)
    text = " ".join(blocks)
    names = character_names(text, labels, max_lower, min_count, min_len)
    masked, n = text, 0
    if names:
        pat = re.compile(r"(?<![A-Za-z])(?i:" + "|".join(sorted(map(re.escape, names), key=len, reverse=True))
                         + r")(?:'s|'S)?(?![A-Za-z])")
        masked, n = pat.subn(name_ph, text)
    return masked, {"unmasked": text, "names": sorted(names), "names_masked": n, "blocks": len(blocks)}


def raw_script(script: str) -> str:
    """The baseline: markup stripped and nothing else, so headings, labels and tokeniser spacing stay."""
    return basic(script)


# ------------------------------------------------------------------ summaries

def clean_summary(raw: str, nlp, title_pat, name_ph: str, title_ph: str) -> tuple[str, dict]:
    t = detokenise(basic(raw))
    t, n_br = BRACKET.subn("", t)
    t, n_paren = PAREN_NAME.subn("", t)
    t, n_title = mask(t, title_pat, title_ph)
    doc = nlp(t)
    out, last, n_people = [], 0, 0
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            out.append(t[last:ent.start_char] + name_ph)
            last, n_people = ent.end_char, n_people + 1
    out.append(t[last:])
    return WS.sub(" ", "".join(out)).strip(), {"brackets": n_br, "actor_parens": n_paren, "titles": n_title,
                                              "people": n_people}


def passages(text: str, size: int, min_last: int) -> list[str]:
    """The text cut into runs of `size` words. A short tail joins nothing and is dropped, unless it is all there is."""
    words = text.split()
    out = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    if len(out) > 1 and len(out[-1].split()) < min_last:
        out.pop()
    return out


# ------------------------------------------------------------------ OCR damage

EDGE = "\"'.,!?;:()[]-*"
OCR_SYMBOL = re.compile(r"[~=|·•^\\{}<>_*#@!:;?\"]")


def junk_share(text: str) -> float:
    """Share of words damaged by OCR: a word that, with its edge punctuation stripped, still mixes letters
    with symbols ("gest:ure", "M~ralda", "we~"). Standalone symbols such as the revision asterisks of
    shooting drafts, and ordinary forms like "T.S.", "10:00", "M-16" or "lounge/club", are not counted."""
    toks = text.split()
    if not toks:
        return 0.0
    bad = 0
    for t in toks:
        core = t.strip(EDGE)
        bad += bool(re.search("[A-Za-z]", t)) and bool(OCR_SYMBOL.search(core or t if re.search("[A-Za-z]", core) else ""))
    return bad / len(toks)
