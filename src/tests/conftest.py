"""Shared setup. Tests use synthetic data only: nothing here reads data/raw, outputs/ or private/.

The scripts/ folders are plain script directories (scripts import each other by bare name, e.g.
pair_returns does `from roi_similarity import ...`), so they go on sys.path the same way.
"""

import os
import re
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# explorer/log.py writes to logs/explorer.log; tests plant crashes, so they log to a throwaway folder instead
os.environ.setdefault("EXPLORER_LOG_DIR", tempfile.mkdtemp(prefix="explorer-test-logs-"))
for p in (ROOT / "src", ROOT / "src" / "scripts" / "network", ROOT / "src" / "scripts" / "screenplays"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def infobox():
    """The helper functions of scripts/imdb/parse_infobox.py. That script parses files as soon as it is
    imported, so only its definitions (everything above the first file-processing line) are executed."""
    src = (ROOT / "src" / "scripts" / "imdb" / "parse_infobox.py").read_text()
    head = src.split("\nIB=re.compile")[0]
    head = re.sub(r"(?m)^SRC=.*$", "", head)  # the argv defaults; not needed for the helpers
    ns = {}
    exec(compile(head, "parse_infobox.py (helpers)", "exec"), ns)
    return ns
