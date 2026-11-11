"""Scattertext-style comparison of Tag Genome language across disagreement groups.

Measures: term rates per 10,000 words in Tag Genome reviews for movies in the
extreme disagreement residual groups (rating_above_money vs rating_below_money).
Misses: name filtering is heuristic (titles + starring), not full NER; residual
groups are sample-specific ranks; reviews stop at 2017.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter

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

GROUP_ABOVE = "rating_above_money"
GROUP_BELOW = "rating_below_money"


def load_audience_corpus(
    imdb_ids: set[str],
    max_per_movie: int = 30,
) -> dict[str, str]:
    """Concatenate up to max_per_movie Tag Genome reviews per movie."""
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
        for piece in re.split(r"[|\n;,]", starring):
            for tok in TOKEN_RE.findall(piece):
                if len(tok) > 2:
                    names.add(tok.lower())
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


def build_group_term_rate_table(
    above_texts: dict[str, str],
    below_texts: dict[str, str],
    ban: set[str],
    min_count: int = 25,
) -> pd.DataFrame:
    """Term rates per 10k words for the two disagreement residual groups."""
    above_counts = count_terms(list(above_texts.values()), ban)
    below_counts = count_terms(list(below_texts.values()), ban)
    ar = rates_per_10k(above_counts)
    br = rates_per_10k(below_counts)
    terms = sorted(
        t
        for t in set(above_counts) & set(below_counts)
        if above_counts[t] >= min_count and below_counts[t] >= min_count
    )
    df = pd.DataFrame(
        {
            "term": terms,
            "rating_above_money_per_10k": [ar[t] for t in terms],
            "rating_below_money_per_10k": [br[t] for t in terms],
            "rating_above_money_n": [above_counts[t] for t in terms],
            "rating_below_money_n": [below_counts[t] for t in terms],
        }
    )
    df["log_above"] = np.log10(df["rating_above_money_per_10k"])
    df["log_below"] = np.log10(df["rating_below_money_per_10k"])
    df["log_ratio_above_over_below"] = df["log_above"] - df["log_below"]
    df["n_movies_above"] = len(above_texts)
    df["n_movies_below"] = len(below_texts)
    return df.sort_values("rating_above_money_n", ascending=False)


def top_term_examples(
    term: str,
    above_texts: dict[str, str],
    below_texts: dict[str, str],
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

    return {GROUP_ABOVE: snip(above_texts), GROUP_BELOW: snip(below_texts)}


def run_disagreement_comparison(
    movies: pd.DataFrame,
    min_count: int = 25,
    max_per_movie: int = 30,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Compare Tag Genome vocabulary for residual disagreement extremes."""
    if "disagreement_group" not in movies.columns:
        raise ValueError("movies must include disagreement_group from disagreement.py")

    above_ids = set(movies.loc[movies["disagreement_group"] == GROUP_ABOVE, "imdb_id"])
    below_ids = set(movies.loc[movies["disagreement_group"] == GROUP_BELOW, "imdb_id"])
    # Prefer films that actually have Tag Genome reviews.
    if "n_reviews_genome" in movies.columns:
        has = set(movies.loc[movies["n_reviews_genome"] > 0, "imdb_id"])
        above_ids &= has
        below_ids &= has

    above_texts = load_audience_corpus(above_ids, max_per_movie=max_per_movie)
    below_texts = load_audience_corpus(below_ids, max_per_movie=max_per_movie)

    ib_path = FEATURES / "wiki_infobox.csv"
    subset = movies[movies["imdb_id"].isin(above_ids | below_ids)].copy()
    if ib_path.exists():
        ib = pd.read_csv(ib_path, usecols=["tconst", "starring", "title"])
        ib = ib.rename(columns={"tconst": "imdb_id"})
        merged = subset[["imdb_id"]].merge(ib, on="imdb_id", how="left")
    else:
        merged = subset.copy()
        merged["starring"] = ""
        if "title" not in merged.columns and "title" in subset.columns:
            merged["title"] = subset["title"].values
    ban = name_blacklist(merged)

    table = build_group_term_rate_table(above_texts, below_texts, ban, min_count=min_count)
    table.to_csv(OUTPUTS / "disagreement_term_rates.csv", index=False)
    meta = pd.Series(
        {
            "group_a": GROUP_ABOVE,
            "group_b": GROUP_BELOW,
            "n_movies_group_a_ids": len(above_ids),
            "n_movies_group_b_ids": len(below_ids),
            "n_movies_with_text_a": len(above_texts),
            "n_movies_with_text_b": len(below_texts),
            "min_term_count": min_count,
            "n_terms": len(table),
            "n_banned_name_tokens": len(ban),
            "max_per_movie": max_per_movie,
        }
    )
    meta.to_csv(OUTPUTS / "scattertext_coverage.csv", header=["value"])
    PRIVATE.mkdir(parents=True, exist_ok=True)
    return table, above_texts, below_texts
