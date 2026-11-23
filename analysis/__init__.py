# Analyses that no longer feed the paper live in archive/analysis/ (see archive/README.md). Adding that folder to
# this package's path keeps their old module names working, e.g. `analysis.attribution`.
from pathlib import Path as _Path

__path__.append(str(_Path(__file__).resolve().parents[1] / "archive" / "analysis"))
