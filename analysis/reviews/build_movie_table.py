"""Build the movie-level analysis table (100k+ clean sample; 10k+ optional).

Measures: theatrical money (gross/budget, log ratio), audience (IMDb rating,
Tag Genome review counts) and critic scores/text length (Wikipedia / Wikidata).
Misses: marketing spend, true studio profit, review dates, post-2017 Tag Genome
coverage, and parser error on Wikipedia money fields.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .paths import DATA, FEATURES, GENOME_ZIP, MOVIE_TABLE, OUTPUTS, WIKI

STREAMING_RE = re.compile(r"netflix|amazon|apple|hulu", re.I)
USD_UNIT = "Q4917"


def _load_movies(tier: str) -> pd.DataFrame:
    path = WIKI / ("movies_100k.csv" if tier == "100k" else "movies_10k.csv")
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "tconst": "imdb_id",
            "primaryTitle": "title",
            "startYear": "year",
            "averageRating": "imdb_rating",
            "votes": "imdb_votes",
        }
    )
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["imdb_rating"] = pd.to_numeric(df["imdb_rating"], errors="coerce")
    df["imdb_votes"] = pd.to_numeric(df["imdb_votes"], errors="coerce")
    df["tier"] = np.where(df["imdb_votes"] >= 100_000, "100k+", "10k-100k")
    return df


def _load_infobox() -> pd.DataFrame:
    ib = pd.read_csv(FEATURES / "wiki_infobox.csv")
    # Drop the Redux duplicate page kept in the raw parse.
    ib = ib[ib["title"] != "Apocalypse Now Redux"].copy()
    ib = ib.rename(columns={"tconst": "imdb_id", "rt_pct": "rt_wiki", "metacritic": "metacritic"})
    for col in ("budget", "gross", "rt_wiki", "metacritic", "critic_chars"):
        ib[col] = pd.to_numeric(ib[col], errors="coerce")
    return ib


def _wikidata_rt() -> pd.Series:
    """Best Rotten Tomatoes percentage per IMDb ID from Wikidata scores."""
    path = FEATURES / "wd_scores.tsv"
    wd = pd.read_csv(path, sep="\t", header=None, names=["imdb", "score", "by", "by_name"], skiprows=1)
    wd["imdb"] = wd["imdb"].astype(str).str.replace('"', "", regex=False)
    wd = wd[wd["by_name"].astype(str).str.contains("Rotten Tomatoes", case=False, na=False)]
    pct = wd["score"].astype(str).str.extract(r"(\d{1,3})\s*%", expand=False)
    wd["rt"] = pd.to_numeric(pct, errors="coerce")
    wd = wd.dropna(subset=["rt"])
    return wd.groupby("imdb")["rt"].max()


def _wikidata_usd_money() -> pd.DataFrame:
    """Max USD budget and gross per IMDb ID from Wikidata (optional 10k tier)."""
    wb = pd.read_csv(
        FEATURES / "wd_budget.tsv",
        sep="\t",
        header=None,
        names=["imdb", "amt", "unit"],
        skiprows=1,
    )
    wg = pd.read_csv(
        FEATURES / "wd_gross.tsv",
        sep="\t",
        header=None,
        names=["imdb", "amt", "unit", "scope"],
        skiprows=1,
    )
    for df in (wb, wg):
        df["imdb"] = df["imdb"].astype(str).str.replace('"', "", regex=False)
        df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    wb = wb[wb["unit"].astype(str).str.contains(USD_UNIT, na=False)]
    wg = wg[wg["unit"].astype(str).str.contains(USD_UNIT, na=False)]
    out = pd.DataFrame(
        {
            "wd_budget": wb.groupby("imdb")["amt"].max(),
            "wd_gross": wg.groupby("imdb")["amt"].max(),
        }
    ).reset_index(names="imdb_id")
    return out


def _reception_chars() -> pd.Series:
    path = FEATURES / "wiki_reception.csv"
    rec = pd.read_csv(path, usecols=["tconst", "reception_found", "reception_chars"])
    rec["reception_chars"] = pd.to_numeric(rec["reception_chars"], errors="coerce").fillna(0)
    found = rec["reception_found"].astype(str).isin(["1", "True", "true"]) | (rec["reception_found"] == 1)
    rec.loc[~found, "reception_chars"] = 0
    return rec.set_index("tconst")["reception_chars"]


def _genome_review_counts(imdb_ids: Iterable[str] | None = None) -> pd.DataFrame:
    """Count Tag Genome reviews and total characters per IMDb ID (no text kept)."""
    import io
    import json
    import zipfile
    from collections import Counter

    want = set(imdb_ids) if imdb_ids is not None else None
    z = zipfile.ZipFile(GENOME_ZIP)
    item2imdb: dict[int, str] = {}
    with z.open("movie_dataset_public_final/raw/metadata.json") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf8"):
            row = json.loads(line)
            if row.get("imdbId") in (None, ""):
                continue
            tt = f"tt{int(row['imdbId']):07d}"
            if want is None or tt in want:
                item2imdb[row["item_id"]] = tt

    n_reviews: Counter[str] = Counter()
    n_chars: Counter[str] = Counter()
    with z.open("movie_dataset_public_final/raw/reviews.json") as fh:
        for line in io.TextIOWrapper(fh, encoding="utf8"):
            row = json.loads(line)
            tt = item2imdb.get(row["item_id"])
            if tt is None:
                continue
            txt = row.get("txt") or ""
            n_reviews[tt] += 1
            n_chars[tt] += len(txt)

    return pd.DataFrame(
        {
            "imdb_id": list(n_reviews.keys()),
            "n_reviews_genome": list(n_reviews.values()),
            "review_chars_genome": [n_chars[k] for k in n_reviews],
        }
    )


def theatrical_clean_mask(df: pd.DataFrame) -> pd.Series:
    """README clean theatrical sample filters (USD, non-streaming, non-regional)."""
    dist = df["distributors"].fillna("").astype(str)
    gflag = df["gross_flag"].fillna("").astype(str)
    streaming = dist.str.contains(STREAMING_RE)
    return (
        (df["budget_cur"] == "USD")
        & (df["gross_cur"] == "USD")
        & (df["budget"] >= 1000)
        & (df["gross"] >= 1000)
        & (~gflag.str.contains("rentals", case=False, na=False))
        & (~gflag.str.contains("regional", case=False, na=False))
        & (~streaming)
        & (df["year"] < 2026)
    )


def add_profit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gross_minus_budget"] = out["gross"] - out["budget"]
    ratio = out["gross"] / out["budget"]
    out["log_gross_over_budget"] = np.log(ratio.replace(0, np.nan))
    out["gross_over_budget"] = ratio
    out["under_budget"] = out["gross"] < out["budget"]
    return out


def build_movie_table(
    max_year: int = 2017,
    include_10k_extension: bool = True,
    genome_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the analysis table. Primary rows are 100k+; 10k+ are flagged optional."""
    m100 = _load_movies("100k")
    ib = _load_infobox()
    base = m100.merge(ib, on="imdb_id", how="left", suffixes=("", "_ib"))
    base["rt_wikidata"] = base["imdb_id"].map(_wikidata_rt())
    base["rotten_tomatoes"] = base["rt_wiki"].fillna(base["rt_wikidata"])
    base["critic_text_chars"] = base["imdb_id"].map(_reception_chars()).fillna(base["critic_chars"]).fillna(0)
    # Prefer dedicated reception extractor length when present.
    base = base[theatrical_clean_mask(base)].copy()
    base["sample"] = "100k+"
    base["money_source"] = "wikipedia_infobox"
    base = add_profit_columns(base)

    frames = [base]

    if include_10k_extension:
        m10 = _load_movies("10k")
        only10 = m10[~m10["imdb_id"].isin(base["imdb_id"])].copy()
        wd_money = _wikidata_usd_money()
        ext = only10.merge(wd_money, on="imdb_id", how="inner")
        ext = ext.rename(columns={"wd_budget": "budget", "wd_gross": "gross"})
        ext = ext[(ext["budget"] >= 1000) & (ext["gross"] >= 1000) & (ext["year"] < 2026)].copy()
        for col in (
            "budget_cur",
            "gross_cur",
            "distributors",
            "gross_flag",
            "metacritic",
            "rotten_tomatoes",
            "critic_text_chars",
            "genres",
        ):
            if col not in ext.columns:
                if col in ("budget_cur", "gross_cur"):
                    ext[col] = "USD"
                elif col == "genres":
                    pass
                else:
                    ext[col] = np.nan
        ext["budget_cur"] = "USD"
        ext["gross_cur"] = "USD"
        ext["sample"] = "10k+_wikidata_optional"
        ext["money_source"] = "wikidata"
        ext = add_profit_columns(ext)
        frames.append(ext)

    table = pd.concat(frames, ignore_index=True, sort=False)

    if genome_counts is None:
        # Count reviews for primary sample through max_year; cheaper than full zip scan twice.
        ids = table.loc[table["year"] <= max_year, "imdb_id"]
        genome_counts = _genome_review_counts(ids)
    table = table.merge(genome_counts, on="imdb_id", how="left")
    table["n_reviews_genome"] = table["n_reviews_genome"].fillna(0).astype(int)
    table["review_chars_genome"] = table["review_chars_genome"].fillna(0).astype(int)

    # Reviews have no date: restrict audience-review analyses to max_year.
    table["reviews_eligible"] = table["year"] <= max_year

    keep = [
        "imdb_id",
        "title",
        "year",
        "genres",
        "tier",
        "sample",
        "money_source",
        "budget",
        "gross",
        "gross_minus_budget",
        "gross_over_budget",
        "log_gross_over_budget",
        "under_budget",
        "imdb_rating",
        "imdb_votes",
        "metacritic",
        "rotten_tomatoes",
        "n_reviews_genome",
        "review_chars_genome",
        "critic_text_chars",
        "reviews_eligible",
    ]
    # title helps quadrant labelling; keep it in the committed summary too.
    out = table[keep].sort_values(["sample", "year", "imdb_id"]).reset_index(drop=True)
    return out


def save_movie_table(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or MOVIE_TABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    # Small coverage summary with numbers from data (safe to commit).
    summary = {
        "n_100k": int((df["sample"] == "100k+").sum()),
        "n_100k_le_2017": int(((df["sample"] == "100k+") & df["reviews_eligible"]).sum()),
        "n_100k_under_budget": int(((df["sample"] == "100k+") & df["under_budget"]).sum()),
        "n_100k_with_genome": int(
            ((df["sample"] == "100k+") & df["reviews_eligible"] & (df["n_reviews_genome"] > 0)).sum()
        ),
        "n_10k_optional": int((df["sample"] == "10k+_wikidata_optional").sum()),
        "n_10k_under_budget": int(
            ((df["sample"] == "10k+_wikidata_optional") & df["under_budget"]).sum()
        ),
    }
    pd.Series(summary).to_csv(OUTPUTS / "movie_table_coverage.csv", header=["value"])
    return path


def primary_sample(df: pd.DataFrame, max_year: int = 2017) -> pd.DataFrame:
    """100k+ clean theatrical sample restricted to movies with review eligibility."""
    return df[(df["sample"] == "100k+") & (df["year"] <= max_year)].copy()


if __name__ == "__main__":
    table = build_movie_table()
    path = save_movie_table(table)
    print(f"wrote {path} rows={len(table)}")
    print(table.groupby("sample").size())
    print("under_budget rates:")
    print(table.groupby("sample")["under_budget"].mean())
