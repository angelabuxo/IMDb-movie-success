"""US consumer prices by year, to put dollars from different years on one footing.

The series is the Bureau of Labor Statistics' CPI-U, all items, US city average, not seasonally adjusted
(1982-84 = 100), taken from FRED as CPIAUCNS. data/cpi_us_annual.csv holds the mean of each year's twelve
monthly values, or of eleven where the BLS published no figure for a month (the months column says which
years). It's a US government series in the public domain. To refresh it, from the repo root:

    python -m analysis.attribution.cpi

What it captures: how much a dollar bought in the US in each year. What it misses: budgets are spent a year or
two before release and we deflate by release year; a worldwide gross is earned in many currencies and countries
with their own inflation; and cinema tickets haven't moved exactly with consumer prices in general.
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "cpi_us_annual.csv"
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS"


def annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Monthly rows (observation_date, CPIAUCNS) to one row per year: the mean of the months published, and how
    many there were. Years with fewer than eleven months are left out (the current year, usually)."""
    m = monthly.assign(year=pd.to_datetime(monthly.observation_date).dt.year, v=pd.to_numeric(monthly.CPIAUCNS, errors="coerce"))
    a = m.groupby("year").v.agg(cpi="mean", months="count").reset_index()
    return a[a.months >= 11].round({"cpi": 3})


def deflator(base_year: int | None = None, path: Path = CSV) -> tuple[pd.Series, int]:
    """A multiplier per year that turns that year's dollars into base-year dollars, and the base year
    (default: the last year in the file)."""
    c = pd.read_csv(path).set_index("year").cpi
    base = int(c.index.max()) if base_year is None else base_year
    return c[base] / c, base


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    raw = Path(src).read_text() if src else urllib.request.urlopen(URL, timeout=60).read().decode()
    a = annual(pd.read_csv(io.StringIO(raw)))
    a.to_csv(CSV, index=False)
    print(f"wrote {CSV}: {a.year.min()} to {a.year.max()}")


if __name__ == "__main__":
    main()
