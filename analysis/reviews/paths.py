"""Paths for the reviews analysis. Raw text stays under build/ (gitignored)."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "build" / "imdb"
FEATURES = DATA / "features"
WIKI = DATA / "wiki"
REVIEWS = DATA / "reviews"
GENOME_ZIP = REVIEWS / "genome" / "genome_2021.zip"

ANALYSIS_DIR = Path(__file__).resolve().parent
OUTPUTS = ANALYSIS_DIR / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Derived tables that may contain review text — never commit these.
PRIVATE = FEATURES / "reviews_private"
PRIVATE.mkdir(parents=True, exist_ok=True)

MOVIE_TABLE = OUTPUTS / "movie_level_table.csv"
HAND_LABEL_CSV = PRIVATE / "hand_label_sample_50.csv"
HAND_LABEL_TEMPLATE = OUTPUTS / "hand_label_template_50.csv"
GENOME_AGG = PRIVATE / "genome_review_agg.parquet"
TEXT_FEATURES = PRIVATE / "movie_text_features.parquet"
