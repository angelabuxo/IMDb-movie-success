"""Paths for the embedding analysis, built from the repo root (data/raw is often a symlink)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("IMDB_DATA_DIR", REPO / "data" / "raw"))
FEATURES = DATA / "features"
MOVIESUM = [DATA / "scripts" / "moviesum" / f"{s}.jsonl" for s in ("train", "val", "test")]
GENOME = DATA / "reviews" / "genome" / "genome_2021.zip"

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"            # committed: numbers and figures only
PRIVATE = HERE / "private"            # gitignored: cleaned review, screenplay and summary text
VECTORS = FEATURES / "text_vectors"   # in data/raw, so ignored: one vector per film, numbers only
for p in (OUTPUTS, PRIVATE):
    p.mkdir(parents=True, exist_ok=True)


def cfg() -> dict:
    return yaml.safe_load((HERE / "config.yaml").read_text())
