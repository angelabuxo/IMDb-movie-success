"""The clean theatrical rule, in one place. Every script, step and notebook that needs the clean sample
calls this module; coverage_clean_sample.sql can't import Python, so a test checks its WHERE clause against
the constants here.

A film is clean when:
  - budget and gross are both in US dollars,
  - budget and gross are both at least $1,000,
  - the gross isn't flagged rentals-only, regional or direct-to-video,
  - no distributor is a streaming service,
  - it was released from 1975 to 2025 (older grosses are mostly distributor rentals).

Input is the movie table written by scripts/imdb (movies_features.csv or movies_features_10k.csv), with the
columns y, b, budget_cur, g, gross_cur, gross_flag and distributors.

What it captures: films whose two money figures are comparable theatrical US-dollar amounts.
What it misses: it trusts the parser's flags, so a film the parser misread can be wrongly kept or dropped,
and gross over budget still isn't profit.
"""
from __future__ import annotations

import re

MIN_BUDGET_USD = 1000
MIN_GROSS_USD = 1000
MIN_YEAR_INCLUSIVE = 1975
MAX_YEAR_EXCLUSIVE = 2026
CURRENCY = "USD"
BAD_GROSS_FLAGS = ("rentals", "regional", "direct_to_video")
STREAMING_DISTRIBUTORS = ("netflix", "amazon", "apple", "hulu")

_FLAGS = "|".join(BAD_GROSS_FLAGS)
_STREAM = re.compile("|".join(STREAMING_DISTRIBUTORS), re.I)

# In the order they are reported: a film gets the first reason that applies.
REASONS = ("no budget or gross parsed", "budget or gross not in USD", f"budget or gross under ${MIN_BUDGET_USD:,}",
           "gross flagged rentals, regional or direct-to-video", "streaming distributor",
           f"released before {MIN_YEAR_INCLUSIVE} or after {MAX_YEAR_EXCLUSIVE - 1}")


def _checks(m):
    """One boolean column per reason a film is NOT clean, in REASONS order."""
    import pandas as pd

    y = pd.to_numeric(m.y, errors="coerce")
    b, g = pd.to_numeric(m.b, errors="coerce"), pd.to_numeric(m.g, errors="coerce")
    return pd.DataFrame({
        REASONS[0]: b.isna() | g.isna(),
        REASONS[1]: (m.budget_cur != CURRENCY) | (m.gross_cur != CURRENCY),
        REASONS[2]: ~(b >= MIN_BUDGET_USD) | ~(g >= MIN_GROSS_USD),
        REASONS[3]: m.gross_flag.fillna("").astype(str).str.contains(_FLAGS),
        REASONS[4]: m.distributors.fillna("").astype(str).str.contains(_STREAM),
        REASONS[5]: ~((y >= MIN_YEAR_INCLUSIVE) & (y < MAX_YEAR_EXCLUSIVE)),
    }, index=m.index)


def clean_mask(m):
    """Boolean Series: True for the clean films of a movie table (a pandas DataFrame)."""
    return ~_checks(m).any(axis=1)


def steps(m):
    """The rule as cumulative steps, in REASONS order: (reason, films that pass that step). Applying them one
    after another gives clean_mask(); the explorer's sample funnel draws them."""
    c = _checks(m)
    return [(r, ~c[r]) for r in REASONS]


def why_not_clean(m):
    """Series: the first reason each film is not clean, or "clean"."""
    c = _checks(m)
    return c.idxmax(axis=1).where(c.any(axis=1), "clean")


def is_clean(r: dict) -> bool:
    """The same rule for one csv.DictReader row (every value a string, missing ones empty)."""
    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    return bool(r.get("budget_cur") == CURRENCY and r.get("gross_cur") == CURRENCY
                and num(r.get("b")) >= MIN_BUDGET_USD and num(r.get("g")) >= MIN_GROSS_USD
                and not re.search(_FLAGS, r.get("gross_flag") or "")
                and not _STREAM.search(r.get("distributors") or "")
                and MIN_YEAR_INCLUSIVE <= num(r.get("y")) < MAX_YEAR_EXCLUSIVE)


def check_config(sample: dict) -> None:
    """The arcs and scripts_text configs record the rule's parameters (they were fixed before results).
    Fail loudly if a config and this module ever disagree."""
    got = {"min_budget_usd": sample["min_budget_usd"], "min_year_inclusive": sample["min_year_inclusive"],
           "max_year_exclusive": sample["max_year_exclusive"],
           "streaming_distributors": sample["streaming_distributors"]}
    want = {"min_budget_usd": MIN_BUDGET_USD, "min_year_inclusive": MIN_YEAR_INCLUSIVE,
            "max_year_exclusive": MAX_YEAR_EXCLUSIVE, "streaming_distributors": "|".join(STREAMING_DISTRIBUTORS)}
    if got != want:
        raise ValueError(f"config.yaml and analysis/clean_rule.py disagree on the clean rule: {got} != {want}")
