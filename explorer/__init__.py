# Explorer pages for archived analyses live in archive/explorer/ (see archive/README.md) and are listed under
# "Archive" in the navigation. Adding that folder to this package's path keeps `from explorer import map_pages` working.
from pathlib import Path as _Path

__path__.append(str(_Path(__file__).resolve().parents[1] / "archive" / "explorer"))
