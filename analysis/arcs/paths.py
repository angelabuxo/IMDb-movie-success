"""Paths for the screenplay arc analysis, all built from the repo root.

data/raw is often a symlink, so nothing here walks up from inside it with "..".
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("IMDB_DATA_DIR", REPO / "data" / "raw"))
FEATURES = DATA / "features"
MOVIESUM = [DATA / "scripts" / "moviesum" / f"{s}.jsonl" for s in ("train", "val", "test")]
EARLIER = REPO / "analysis" / "scripts_text" / "outputs"   # committed outputs of the Scattertext branch

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
OUTPUTS = HERE / "outputs"          # committed: aggregate tables, arc curves (numbers only), figures
PRIVATE = HERE / "private"          # gitignored: script text, IMDb fields, sentence scores
SITE = REPO / "site" / "arcs"
NOTEBOOKS = REPO / "notebooks" / "arcs"
for p in (OUTPUTS, PRIVATE, SITE):
    p.mkdir(parents=True, exist_ok=True)
