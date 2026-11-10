"""Audience review text features (Tag Genome) and sentiment checks.

Measures: review length, type-token ratio on fixed-size token samples, and
VADER compound sentiment aggregated to movies / genres.
Misses: VADER is a general social-media lexicon (not movie-specific); Tag Genome
reviews have no date or star rating; fixed-size sampling discards short reviews.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .paths import (
    GENOME_ZIP,
    HAND_LABEL_CSV,
    HAND_LABEL_TEMPLATE,
    OUTPUTS,
    PRIVATE,
    REVIEWS,
    TEXT_FEATURES,
)

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def lexical_diversity(tokens: list[str], sample_size: int = 100, random_state: int = 0) -> float:
    """Type-token ratio on a fixed-size sample so length does not drive diversity."""
    if len(tokens) < sample_size:
        return np.nan
    rng = np.random.default_rng(random_state)
    sample = rng.choice(tokens, size=sample_size, replace=False)
    return len(set(sample)) / sample_size


def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def iter_genome_reviews(
    imdb_ids: set[str] | None = None,
    max_per_movie: int | None = None,
) -> Iterator[tuple[str, str, int]]:
    """Yield (imdb_id, text, item_id) from Tag Genome. Does not print text."""
    z = zipfile.ZipFile(GENOME_ZIP)
    item2imdb: dict[int, str] = {}
    with z.open("movie_dataset_public_final/raw/metadata.json") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf8"):
            row = json.loads(line)
            if row.get("imdbId") in (None, ""):
                continue
            tt = f"tt{int(row['imdbId']):07d}"
            if imdb_ids is None or tt in imdb_ids:
                item2imdb[row["item_id"]] = tt

    per: dict[str, int] = defaultdict(int)
    with z.open("movie_dataset_public_final/raw/reviews.json") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf8"):
            row = json.loads(line)
            tt = item2imdb.get(row["item_id"])
            if tt is None:
                continue
            if max_per_movie is not None and per[tt] >= max_per_movie:
                continue
            txt = row.get("txt") or ""
            if not txt.strip():
                continue
            per[tt] += 1
            yield tt, txt, int(row["item_id"])


def compute_movie_text_features(
    imdb_ids: set[str],
    sample_size: int = 100,
    max_per_movie: int = 50,
    random_state: int = 0,
) -> pd.DataFrame:
    """Aggregate length, diversity and VADER sentiment per movie."""
    analyzer = _vader()
    lengths: dict[str, list[int]] = defaultdict(list)
    diversities: dict[str, list[float]] = defaultdict(list)
    compounds: dict[str, list[float]] = defaultdict(list)
    n_used: dict[str, int] = defaultdict(int)

    for i, (tt, txt, _) in enumerate(iter_genome_reviews(imdb_ids, max_per_movie=max_per_movie)):
        tokens = tokenize(txt)
        lengths[tt].append(len(tokens))
        div = lexical_diversity(tokens, sample_size=sample_size, random_state=random_state + i)
        if not np.isnan(div):
            diversities[tt].append(div)
        compounds[tt].append(analyzer.polarity_scores(txt)["compound"])
        n_used[tt] += 1

    rows = []
    for tt in sorted(imdb_ids):
        if n_used[tt] == 0:
            continue
        rows.append(
            {
                "imdb_id": tt,
                "n_reviews_used": n_used[tt],
                "mean_review_tokens": float(np.mean(lengths[tt])),
                "median_review_tokens": float(np.median(lengths[tt])),
                "mean_lexical_diversity": float(np.mean(diversities[tt])) if diversities[tt] else np.nan,
                "mean_vader_compound": float(np.mean(compounds[tt])),
                "median_vader_compound": float(np.median(compounds[tt])),
                "frac_vader_positive": float(np.mean(np.array(compounds[tt]) >= 0.05)),
                "frac_vader_negative": float(np.mean(np.array(compounds[tt]) <= -0.05)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_parquet(TEXT_FEATURES, index=False)
    return out


def genre_sentiment_table(movies: pd.DataFrame, text_features: pd.DataFrame) -> pd.DataFrame:
    """Explode genres and summarise VADER / diversity (movie counted in each genre)."""
    m = movies.merge(text_features, on="imdb_id", how="inner")
    m = m.dropna(subset=["genres"])
    rows = []
    for _, row in m.iterrows():
        for g in str(row["genres"]).split(","):
            g = g.strip()
            if not g:
                continue
            rows.append(
                {
                    "genre": g,
                    "imdb_id": row["imdb_id"],
                    "mean_vader_compound": row["mean_vader_compound"],
                    "mean_lexical_diversity": row["mean_lexical_diversity"],
                    "mean_review_tokens": row["mean_review_tokens"],
                    "log_gross_over_budget": row.get("log_gross_over_budget", np.nan),
                    "imdb_rating": row.get("imdb_rating", np.nan),
                }
            )
    long = pd.DataFrame(rows)
    summary = (
        long.groupby("genre")
        .agg(
            n_movies=("imdb_id", "nunique"),
            mean_vader=("mean_vader_compound", "mean"),
            mean_diversity=("mean_lexical_diversity", "mean"),
            mean_tokens=("mean_review_tokens", "mean"),
            mean_log_profit=("log_gross_over_budget", "mean"),
            mean_imdb=("imdb_rating", "mean"),
        )
        .reset_index()
        .sort_values("n_movies", ascending=False)
    )
    return summary


def export_hand_label_sample(
    imdb_ids: set[str],
    n: int = 50,
    random_state: int = 42,
) -> tuple[Path, Path]:
    """Export n random reviews for hand labelling (text file is gitignored)."""
    pool = []
    for tt, txt, item_id in iter_genome_reviews(imdb_ids, max_per_movie=5):
        pool.append((tt, item_id, txt))
        if len(pool) >= 20_000:
            break
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    analyzer = _vader()
    rows = []
    for i, j in enumerate(idx, start=1):
        tt, item_id, txt = pool[int(j)]
        compound = analyzer.polarity_scores(txt)["compound"]
        auto = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
        rows.append(
            {
                "sample_id": i,
                "imdb_id": tt,
                "genome_item_id": item_id,
                "vader_compound": compound,
                "vader_label": auto,
                "human_label": "",  # fill with positive / negative / neutral
                "review_text": txt,
            }
        )
    full = pd.DataFrame(rows)
    full.to_csv(HAND_LABEL_CSV, index=False)
    # Template without review text — safe to commit; join locally on sample_id.
    template = full.drop(columns=["review_text"])
    template.to_csv(HAND_LABEL_TEMPLATE, index=False)
    return HAND_LABEL_CSV, HAND_LABEL_TEMPLATE


def agreement_from_hand_labels(path: Path | None = None) -> dict:
    """Compute agreement once human_label is filled in the private CSV."""
    path = path or HAND_LABEL_CSV
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    df = pd.read_csv(path)
    labelled = df[df["human_label"].astype(str).str.strip().str.len() > 0].copy()
    if labelled.empty:
        return {
            "status": "awaiting_labels",
            "n_exported": int(len(df)),
            "n_labelled": 0,
            "path": str(path),
        }
    labelled["human_label"] = labelled["human_label"].str.strip().str.lower()
    labelled["vader_label"] = labelled["vader_label"].str.strip().str.lower()
    agree = (labelled["human_label"] == labelled["vader_label"]).mean()
    # Binary: collapse neutral with the auto side for a second view.
    both_pn = labelled[labelled["human_label"].isin(["positive", "negative"])]
    binary = np.nan
    if len(both_pn):
        binary = (both_pn["human_label"] == both_pn["vader_label"]).mean()
    return {
        "status": "ok",
        "n_labelled": int(len(labelled)),
        "accuracy": float(agree),
        "accuracy_pos_neg_only": float(binary) if binary == binary else None,
        "path": str(path),
    }


def validate_vader_on_stanford(max_docs: int = 2000, random_state: int = 0) -> dict:
    """Second check: VADER vs Stanford Large Movie Review polarity labels.

    Measures: agreement of VADER's sign with gold pos/neg labels on a sample.
    Misses: those reviews are not Tag Genome text, so transfer is imperfect.
    """
    # Prefer the unpacked aclImdb tree if present; else HF parquet.
    acl = REVIEWS / "stanford" / "aclImdb"
    texts: list[str] = []
    labels: list[int] = []
    if (acl / "test" / "pos").exists():
        for lab, name in ((1, "pos"), (0, "neg")):
            folder = acl / "test" / name
            files = sorted(folder.glob("*.txt"))[: max_docs // 2]
            for fp in files:
                texts.append(fp.read_text(encoding="utf8", errors="ignore"))
                labels.append(lab)
    else:
        train = pd.read_parquet(REVIEWS / "hf" / "train.parquet")
        rng = np.random.default_rng(random_state)
        take = rng.choice(len(train), size=min(max_docs, len(train)), replace=False)
        sample = train.iloc[take]
        texts = sample["text"].astype(str).tolist()
        labels = sample["label"].astype(int).tolist()

    analyzer = _vader()
    pred = []
    for t in texts:
        c = analyzer.polarity_scores(t)["compound"]
        pred.append(1 if c >= 0.05 else (0 if c <= -0.05 else -1))
    # Drop VADER neutrals for binary accuracy against gold pos/neg.
    pairs = [(p, y) for p, y in zip(pred, labels) if p in (0, 1)]
    if not pairs:
        return {"n": 0, "accuracy": np.nan}
    acc = np.mean([p == y for p, y in pairs])
    return {
        "n_docs": len(texts),
        "n_non_neutral": len(pairs),
        "accuracy_vs_stanford": float(acc),
        "frac_vader_neutral": float(np.mean(np.array(pred) == -1)),
    }


def sentiment_profit_by_genre_null(
    movies: pd.DataFrame,
    text_features: pd.DataFrame,
    n_perm: int = 500,
    random_state: int = 0,
) -> pd.DataFrame:
    """Per-genre Spearman(sentiment, log profit) vs decade-stratified null."""
    from .nulls import decade_label, spearman_with_null

    m = movies.merge(text_features, on="imdb_id", how="inner")
    rows = []
    for genre, sub in _explode_genre(m).groupby("genre"):
        if len(sub) < 30:
            continue
        # Within one genre, stratify the null by decade only.
        strata = decade_label(sub["year"])
        res = spearman_with_null(
            sub["mean_vader_compound"],
            sub["log_gross_over_budget"],
            n_perm=n_perm,
            random_state=random_state,
            strata=strata,
        )
        rows.append({"genre": genre, **res})
    return pd.DataFrame(rows).sort_values("rho")


def _explode_genre(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for g in str(row.get("genres") or "").split(","):
            g = g.strip()
            if g:
                d = row.to_dict()
                d["genre"] = g
                rows.append(d)
    return pd.DataFrame(rows)
