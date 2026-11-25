"""Embed the plot section of each film's Wikipedia article.

Run from the data folder (data/raw) after the Wikipedia pull:

    cd data/raw && ../../.venv/bin/python ../../scripts/reviews/plot_embeddings.py

A plot summary says what happens in the film. It was written after release,
like everything on Wikipedia, but it doesn't say how the film was received or
how many people saw it, which is what leaks into the Tag Genome scores.

For every article in wiki/articles.jsonl and wiki/articles_10k.jsonl it takes
the first level-2 section whose heading is one of HEADINGS, strips the wiki
markup, splits the text into passages of at most MAX_WORDS words, embeds them
with all-MiniLM-L6-v2 (the model used for the screenplay and review vectors)
and writes each film's mean vector, normalised, to features/plot_vectors.npy.
features/plot_vector_ids.csv has the row order with the heading used and the
word and passage counts. Plot text is never written out.

What it misses: 180-word passages averaged into one vector keep topic and
setting, not structure or quality. Longer plots belong to better-known films,
so the word count is kept for a check.
"""

import csv
import json
import re

import mwparserfromhell
import numpy as np
from sentence_transformers import SentenceTransformer

SOURCES = ["wiki/articles.jsonl", "wiki/articles_10k.jsonl"]
OUT = "features/plot_vectors.npy"
OUT_IDS = "features/plot_vector_ids.csv"
MODEL = "all-MiniLM-L6-v2"
MAX_WORDS = 180
MIN_WORDS = 40
HEADINGS = ("plot", "plot summary", "plot synopsis", "synopsis", "premise", "summary", "story", "storyline", "plot overview")
SECTION = re.compile(r"^==\s*([^=].*?)\s*==\s*$", re.M)


def plot_section(wikitext):
    """(heading, plain text) of the first plot-like level-2 section, or (None, "")."""
    heads = list(SECTION.finditer(wikitext))
    for k, h in enumerate(heads):
        name = re.sub(r"[^a-z ]", "", h.group(1).lower()).strip()
        if name in HEADINGS:
            end = heads[k + 1].start() if k + 1 < len(heads) else len(wikitext)
            raw = re.sub(r"<ref[^>]*?/>|<ref.*?</ref>", "", wikitext[h.end():end], flags=re.S)
            text = mwparserfromhell.parse(raw).strip_code(normalize=True, collapse=True)
            return name, re.sub(r"\s+", " ", text).strip()
    return None, ""


def passages(text):
    words = text.split()
    return [" ".join(words[i:i + MAX_WORDS]) for i in range(0, len(words), MAX_WORDS)]


def main():
    films, seen, no_plot, short = [], set(), 0, 0
    for src in SOURCES:
        with open(src, encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                ids = [t for t in (r.get("tconst") or []) if t not in seen]
                if not ids or not r.get("wikitext"):
                    continue
                head, text = plot_section(r["wikitext"])
                n = len(text.split())
                if head is None:
                    no_plot += 1
                    continue
                if n < MIN_WORDS:
                    short += 1
                    continue
                for t in ids:
                    seen.add(t)
                    films.append((t, head, n, passages(text)))
    print(f"{len(films):,} films with a plot section of {MIN_WORDS}+ words; {no_plot:,} articles without one, {short:,} too short")
    model = SentenceTransformer(MODEL)
    flat = [p for f in films for p in f[3]]
    vec = model.encode(flat, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
    out, k = np.empty((len(films), vec.shape[1]), dtype=np.float32), 0
    for i, f in enumerate(films):
        v = vec[k:k + len(f[3])].mean(axis=0)
        out[i] = v / np.linalg.norm(v)
        k += len(f[3])
    np.save(OUT, out)
    with open(OUT_IDS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tconst", "heading", "words", "passages"])
        for t, head, n, p in films:
            w.writerow([t, head, n, len(p)])
    words = np.array([f[2] for f in films])
    print(f"wrote {OUT}: {out.shape}; {len(flat):,} passages; words per plot median {int(np.median(words))}, "
          f"10th to 90th percentile {int(np.percentile(words, 10))} to {int(np.percentile(words, 90))}")
    heads = {}
    for f in films:
        heads[f[1]] = heads.get(f[1], 0) + 1
    print("headings used:", dict(sorted(heads.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
