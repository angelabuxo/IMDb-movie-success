"""Scattertext-style comparison of critic vs audience language.

Measures: term rates per 10,000 words in Wikipedia reception text vs Tag Genome
reviews; points near the diagonal share vocabulary.
Misses: reception sections mix quotes and editorial voice; audience reviews are
pre-2018 only; name filtering is heuristic (starring lists + title tokens), not
full NER.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import FEATURES, GENOME_ZIP, OUTPUTS, PRIVATE
from .text_features import TOKEN_RE, tokenize

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "as", "by", "with", "from", "into", "about", "than", "then", "that",
    "this", "these", "those", "it", "its", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "not", "no", "nor", "so",
    "such", "too", "very", "just", "also", "more", "most", "some", "any", "all",
    "each", "few", "other", "over", "after", "before", "between", "out", "up",
    "down", "he", "she", "they", "them", "his", "her", "their", "we", "you",
    "i", "me", "my", "our", "your", "who", "whom", "which", "what", "when",
    "where", "why", "how", "there", "here", "film", "movie", "films", "movies",
    "one", "two", "three", "said", "says", "like", "ref", "cite", "http", "https",
    "www", "com", "org",
}


def load_reception_text(imdb_ids: set[str]) -> dict[str, str]:
    """Load Wikipedia reception plain text for selected movies."""
    path = FEATURES / "wiki_reception.csv"
    # reception_text can contain newlines; pandas handles quoted CSV.
    usecols = ["tconst", "reception_found", "reception_text"]
    df = pd.read_csv(path, usecols=usecols)
    found = df["reception_found"].astype(str).isin(["1", "True", "true"]) | (df["reception_found"] == 1)
    df = df[found & df["tconst"].isin(imdb_ids)]
    out = {}
    for _, row in df.iterrows():
        text = row["reception_text"]
        if isinstance(text, str) and text.strip():
            out[row["tconst"]] = text
    return out


def load_audience_corpus(
    imdb_ids: set[str],
    max_per_movie: int = 30,
) -> dict[str, str]:
    """Concatenate up to max_per_movie Tag Genome reviews per movie.

    Stops once every requested movie has max_per_movie reviews (or the zip ends),
    so coverage is not truncated by a global character budget.
    """
    z = zipfile.ZipFile(GENOME_ZIP)
    item2imdb: dict[int, str] = {}
    with z.open("movie_dataset_public_final/raw/metadata.json") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf8"):
            row = json.loads(line)
            if row.get("imdbId") in (None, ""):
                continue
            tt = f"tt{int(row['imdbId']):07d}"
            if tt in imdb_ids:
                item2imdb[row["item_id"]] = tt

    buckets: dict[str, list[str]] = {tt: [] for tt in imdb_ids}
    counts: Counter[str] = Counter()
    done = 0
    target = len(imdb_ids)
    with z.open("movie_dataset_public_final/raw/reviews.json") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf8"):
            if done >= target:
                break
            row = json.loads(line)
            tt = item2imdb.get(row["item_id"])
            if tt is None or counts[tt] >= max_per_movie:
                continue
            txt = (row.get("txt") or "").strip()
            if not txt:
                continue
            buckets[tt].append(txt)
            counts[tt] += 1
            if counts[tt] == max_per_movie:
                done += 1
    return {tt: "\n".join(parts) for tt, parts in buckets.items() if parts}


def name_blacklist(movies: pd.DataFrame) -> set[str]:
    """Person/character-ish tokens from titles and starring lists."""
    names: set[str] = set()
    for _, row in movies.iterrows():
        title = str(row.get("title") or "")
        for tok in TOKEN_RE.findall(title):
            if len(tok) > 2:
                names.add(tok.lower())
        starring = str(row.get("starring") or "")
        # Infobox starring may not be on the analysis table; optional column.
        for piece in re.split(r"[|\n;,]", starring):
            for tok in TOKEN_RE.findall(piece):
                if len(tok) > 2:
                    names.add(tok.lower())
    # Common given names that dominate movie text without being informative here.
    names.update(
        {
            "john", "james", "david", "michael", "robert", "william", "mary",
            "jack", "tom", "tony", "peter", "paul", "george", "chris", "steve",
        }
    )
    return names


def count_terms(texts: list[str], ban: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for tok in tokenize(text):
            if tok in STOPWORDS or tok in ban or len(tok) < 3:
                continue
            counts[tok] += 1
    return counts


def rates_per_10k(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values()) or 1
    return {t: 10000.0 * c / total for t, c in counts.items()}


def build_term_rate_table(
    critic_texts: dict[str, str],
    audience_texts: dict[str, str],
    ban: set[str],
    min_count: int = 25,
) -> pd.DataFrame:
    """Term rates per 10k words for critic and audience corpora (shared movies)."""
    shared = sorted(set(critic_texts) & set(audience_texts))
    critic_counts = count_terms([critic_texts[tt] for tt in shared], ban)
    audience_counts = count_terms([audience_texts[tt] for tt in shared], ban)
    cr = rates_per_10k(critic_counts)
    ar = rates_per_10k(audience_counts)
    # Keep terms frequent enough in BOTH corpora so log rates are comparable.
    terms = sorted(
        t
        for t in set(critic_counts) & set(audience_counts)
        if critic_counts[t] >= min_count and audience_counts[t] >= min_count
    )
    df = pd.DataFrame(
        {
            "term": terms,
            "critic_per_10k": [cr[t] for t in terms],
            "audience_per_10k": [ar[t] for t in terms],
            "critic_n": [critic_counts[t] for t in terms],
            "audience_n": [audience_counts[t] for t in terms],
        }
    )
    df["log_critic"] = np.log10(df["critic_per_10k"])
    df["log_audience"] = np.log10(df["audience_per_10k"])
    df["log_ratio_audience_over_critic"] = df["log_audience"] - df["log_critic"]
    df["n_movies_shared"] = len(shared)
    return df.sort_values("audience_n", ascending=False)


def top_term_examples(
    term: str,
    critic_texts: dict[str, str],
    audience_texts: dict[str, str],
    n: int = 3,
    window: int = 80,
) -> dict[str, list[str]]:
    """Short windows around a term for inspection (not for publishing long quotes)."""
    pat = re.compile(rf"\b{re.escape(term)}\b", re.I)

    def snip(corpus: dict[str, str]) -> list[str]:
        out = []
        for tt, text in corpus.items():
            m = pat.search(text or "")
            if not m:
                continue
            a = max(0, m.start() - window)
            b = min(len(text), m.end() + window)
            out.append(f"{tt}: …{text[a:b].replace(chr(10), ' ')}…")
            if len(out) >= n:
                break
        return out

    return {"critic": snip(critic_texts), "audience": snip(audience_texts)}


def run_comparison(
    movies: pd.DataFrame,
    min_count: int = 25,
    max_per_movie: int = 30,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    ids = set(movies["imdb_id"])
    critic = load_reception_text(ids)
    audience = load_audience_corpus(ids, max_per_movie=max_per_movie)
    # Starring names if available from infobox file.
    ib_path = FEATURES / "wiki_infobox.csv"
    if ib_path.exists():
        ib = pd.read_csv(ib_path, usecols=["tconst", "starring", "title"])
        ib = ib.rename(columns={"tconst": "imdb_id"})
        merged = movies[["imdb_id"]].merge(ib, on="imdb_id", how="left")
    else:
        merged = movies.copy()
        merged["starring"] = ""
    ban = name_blacklist(merged)
    table = build_term_rate_table(critic, audience, ban, min_count=min_count)
    out_path = OUTPUTS / "critic_vs_audience_term_rates.csv"
    # Rates only — no raw text.
    table.to_csv(out_path, index=False)
    meta = pd.Series(
        {
            "n_movies_with_critic_text": len(critic),
            "n_movies_with_audience_text": len(audience),
            "n_shared": len(set(critic) & set(audience)),
            "min_term_count": min_count,
            "n_terms": len(table),
            "n_banned_name_tokens": len(ban),
        }
    )
    meta.to_csv(OUTPUTS / "scattertext_coverage.csv", header=["value"])
    # Keep example windows private (may contain copyrighted review text).
    PRIVATE.mkdir(parents=True, exist_ok=True)
    return table, critic, audience
