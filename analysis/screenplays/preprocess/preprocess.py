"""Preprocessing for the Word2Vec analysis (DECISIONS.md, decided table, "Preprocessing").

One tokeniser for both layers:
  * lowercase;
  * curly apostrophes become straight ones, and a detached contraction is joined to the word before it
    ("do n't" -> "don't", "she 's" -> "she's");
  * every symbol that is not a letter is removed (digits too: they are not letters). A token is a run of letters,
    with an apostrophe allowed inside a word. A hyphenated word is split in two;
  * no lemmatisation and no stopword removal;
  * the film's own character names are removed from both layers. The list comes from the text of that film's
    <character> elements (parenthesised extensions "(V.O.)" dropped, the rest tokenised), MINUS the tokens that are a
    character-label token in NAME_MAX_FILMS or more films of the 2188. Those are mostly ordinary words used as labels
    (man, voice, guard, "the"), not names of one film, and removing them would delete the word wherever the label
    is used; the IDF weighting already reduces the names that are shared by many films.
    NAME_MAX_FILMS = 20 is arbitrary (there is no natural cut) and was fixed before any AUC was computed; the only
    criterion is "the token is specific to a few films".

Context unit: each element (one line of dialogue, one description) is its own list of tokens, so a Word2Vec
window never crosses elements. Only the dialogue and description layers are kept; parentheticals, characters and
stage directions are out of both.

Functions work on strings and lists, so they are testable without the data (tests/test_screenplays_preprocess.py).

Run (numbers and a before/after sample, no corpus written):
    python3 -I analysis/screenplays/preprocess/preprocess.py
Writes preprocess_summary.csv (numbers only) here and private/preprocess_sample.csv (gitignored, text of at most
15 words per row) for a manual look.
"""
import csv
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "data_check"))
sys.path.insert(0, str(HERE.parent / "genre_counts"))

LAYERS = ("dialogue", "description")
DETACHED = re.compile(r"(\w)\s+(n't|'s|'ll|'re|'ve|'d|'m)\b", re.IGNORECASE)
TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")  # letters, an apostrophe only between letters
PAREN = re.compile(r"\([^)]*\)")
MAX_WORDS = 15
NAME_MAX_FILMS = 20  # a label token in this many films or more is a common word, not a name


def normalise(text):
    """Lowercase, straight apostrophes, detached contractions joined to the word before them."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    return DETACHED.sub(r"\1\2", text)


def tokenize(text):
    return TOKEN.findall(normalise(text))


def name_tokens(character_texts):
    """Set of tokens that appear in the film's <character> elements (parenthesised extensions dropped)."""
    names = set()
    for t in character_texts:
        names.update(tokenize(PAREN.sub(" ", t)))
    return names


def label_tokens(elements):
    """Tokens of the <character> elements of one film (before the common-word filter)."""
    return name_tokens(e["text"] for e in elements if e["tag"] == "character" and e["text"])


def film_layers(elements, common_labels=frozenset()):
    """elements: parse_layers.parse_script output. Returns (layers, names, stats).

    names = the film's label tokens minus common_labels (label tokens found in NAME_MAX_FILMS or more films).

    layers: {"dialogue": [[token, ...], ...], "description": [...]}, one list per non-empty element.
    stats: tokens before the name removal and tokens removed as names, per layer.
    """
    names = label_tokens(elements) - set(common_labels)
    layers = {k: [] for k in LAYERS}
    stats = {k: {"before": 0, "names_removed": 0} for k in LAYERS}
    for e in elements:
        if e["layer"] not in LAYERS or not e["text"]:
            continue
        toks = tokenize(e["text"])
        kept = [t for t in toks if t not in names]
        stats[e["layer"]]["before"] += len(toks)
        stats[e["layer"]]["names_removed"] += len(toks) - len(kept)
        if kept:
            layers[e["layer"]].append(kept)
    return layers, names, stats


def short(text, n=MAX_WORDS):
    w = text.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def main():
    from genre_counts import DATA
    from parse_layers import one_per_id, parse_script, read_records

    records = one_per_id(read_records(DATA))
    parsed = [(rec, parse_script(rec["script"])) for rec in records]
    label_df = Counter()  # in how many films a token is a character-label token
    for _, els in parsed:
        label_df.update(label_tokens(els))
    common = {t for t, c in label_df.items() if c >= NAME_MAX_FILMS}
    rng = random.Random(2805)
    tot = {k: Counter() for k in LAYERS}
    films_with_empty = 0
    names_per_film = []
    sample_rows = []
    for rec, els in parsed:
        layers, names, stats = film_layers(els, common)
        names_per_film.append(len(names))
        for k in LAYERS:
            tot[k]["raw_whitespace_tokens"] += sum(len(e["text"].split()) for e in els if e["layer"] == k and e["text"])
            tot[k]["tokens_before_names"] += stats[k]["before"]
            tot[k]["names_removed"] += stats[k]["names_removed"]
            tot[k]["tokens_after"] += sum(len(x) for x in layers[k])
            tot[k]["elements_kept"] += len(layers[k])
        films_with_empty += not layers["dialogue"] or not layers["description"]
        for k in LAYERS:  # a few raw elements for a manual before/after look
            cand = [e for e in els if e["layer"] == k and len(e["text"].split()) >= 3]
            for e in rng.sample(cand, min(1, len(cand))) if rng.random() < 0.02 else []:
                sample_rows.append({"imdb_id": rec["imdb_id"], "layer": k, "before": short(e["text"]),
                                    "after": short(" ".join(t for t in tokenize(e["text"]) if t not in names))})

    rows = [("films", len(records)), ("films_with_an_empty_layer_after_preprocessing", films_with_empty),
            ("name_max_films", NAME_MAX_FILMS), ("label_token_types", len(label_df)),
            ("label_token_types_kept_as_common_words", len(common)),
            ("median_name_tokens_per_film", sorted(names_per_film)[len(names_per_film) // 2])]
    for k in LAYERS:
        for stat, v in tot[k].items():
            rows.append((f"{k}_{stat}", v))
        rows.append((f"{k}_share_of_tokens_removed_as_names",
                     round(tot[k]["names_removed"] / max(1, tot[k]["tokens_before_names"]), 4)))
    with open(HERE / "preprocess_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "value"])
        w.writerows(rows)
    private = HERE.parent / "private"
    private.mkdir(exist_ok=True)
    with open(private / "preprocess_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["imdb_id", "layer", "before", "after"])
        w.writeheader()
        w.writerows(sample_rows[:60])
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()
