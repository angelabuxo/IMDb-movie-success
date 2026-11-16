"""The shared clean theatrical rule (analysis/clean_rule.py): each condition, the two entry points agreeing,
and the places that can't import it (the SQL file, the two config.yaml files) staying in step."""

import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from analysis import clean_rule as cr

ROOT = Path(__file__).resolve().parents[1]
GOOD = {"y": 2005, "b": 5e6, "budget_cur": "USD", "g": 2e7, "gross_cur": "USD", "gross_flag": None,
        "distributors": "Warner Bros."}
CASES = [
    ({}, "clean"),
    ({"b": None}, cr.REASONS[0]), ({"g": None}, cr.REASONS[0]),
    ({"budget_cur": "EUR"}, cr.REASONS[1]), ({"gross_cur": "INR"}, cr.REASONS[1]),
    ({"b": 999}, cr.REASONS[2]), ({"g": 999}, cr.REASONS[2]), ({"b": 1000, "g": 1000}, "clean"),
    ({"gross_flag": "rentals"}, cr.REASONS[3]), ({"gross_flag": "rentals;regional"}, cr.REASONS[3]),
    ({"gross_flag": "direct_to_video"}, cr.REASONS[3]), ({"gross_flag": "net_gross"}, "clean"),
    ({"distributors": "Netflix"}, cr.REASONS[4]), ({"distributors": "A24|Apple TV+"}, cr.REASONS[4]),
    ({"distributors": "Amazon MGM Studios"}, cr.REASONS[4]), ({"distributors": None}, "clean"),
    ({"y": 1974}, cr.REASONS[5]), ({"y": 1975}, "clean"), ({"y": 2025}, "clean"), ({"y": 2026}, cr.REASONS[5]),
    ({"y": None}, cr.REASONS[5]),
]


def as_csv_row(r: dict) -> dict:
    """What csv.DictReader gives: every value a string, missing ones empty."""
    return {k: "" if v is None else str(v) for k, v in r.items()}


@pytest.mark.parametrize("change,reason", CASES, ids=[f"{c}" for c, _ in CASES])
def test_each_condition(change, reason):
    row = {**GOOD, **change}
    m = pd.DataFrame([row])
    assert cr.why_not_clean(m).iloc[0] == reason
    assert bool(cr.clean_mask(m).iloc[0]) == (reason == "clean")
    assert cr.is_clean(as_csv_row(row)) == (reason == "clean")


def test_table_and_row_versions_agree_on_a_mixed_table():
    m = pd.DataFrame([{**GOOD, **c} for c, _ in CASES])
    assert list(cr.clean_mask(m)) == [cr.is_clean(as_csv_row(r)) for r in m.to_dict("records")]
    assert int(cr.clean_mask(m).sum()) == sum(r == "clean" for _, r in CASES)


def test_first_reason_wins():
    m = pd.DataFrame([{**GOOD, "budget_cur": "EUR", "y": 1960, "gross_flag": "rentals"}])
    assert cr.why_not_clean(m).iloc[0] == cr.REASONS[1]


def test_sql_copy_matches_the_module():
    """coverage_clean_sample.sql can't import the module, so its WHERE clause is checked against it."""
    sql = (ROOT / "scripts" / "imdb" / "coverage_clean_sample.sql").read_text()
    where = next(l for l in sql.splitlines() if l.startswith("CREATE TEMP TABLE cl AS"))
    assert f"budget_cur='{cr.CURRENCY}'" in where and f"gross_cur='{cr.CURRENCY}'" in where
    assert f"b>={cr.MIN_BUDGET_USD}" in where and f"g>={cr.MIN_GROSS_USD}" in where
    assert f"y>={cr.MIN_YEAR_INCLUSIVE}" in where and f"y<{cr.MAX_YEAR_EXCLUSIVE}" in where
    assert set(re.findall(r"NOT LIKE '%(\w+)%'", where)) == set(cr.BAD_GROSS_FLAGS)
    assert set(re.findall(r"ILIKE '%(\w+)%'", where)) == set(cr.STREAMING_DISTRIBUTORS)


@pytest.mark.parametrize("pkg", ["arcs", "scripts_text"])
def test_configs_match_the_module(pkg):
    sample = yaml.safe_load((ROOT / "analysis" / pkg / "config.yaml").read_text())["sample"]
    cr.check_config(sample)
    with pytest.raises(ValueError):
        cr.check_config({**sample, "min_year_inclusive": 1900})


def test_no_other_copy_of_the_rule():
    """The rule's tell-tale code (the flag and streamer patterns) appears in no script, step or notebook but the shared module and the SQL."""
    allowed = {"analysis/clean_rule.py", "scripts/imdb/coverage_clean_sample.sql", "scripts/imdb/parse_infobox.py",
               "tests/test_clean_rule.py", "tests/test_network_and_parsing.py",
               "analysis/arcs/x02_clean_rule_gap.py"}   # rebuilds the 14 November rule on purpose
    hits = []
    for pat in ("analysis/**/*.py", "scripts/**/*.py", "explorer/*.py", "notebooks/**/*.ipynb"):
        for p in ROOT.glob(pat):
            rel = p.relative_to(ROOT).as_posix()
            if rel not in allowed and re.search(r"rentals\|regional|netflix\|amazon|\"regional\\?\" not in", p.read_text()):
                hits.append(rel)
    assert not hits, f"the clean rule is written out again in {hits}; call analysis.clean_rule instead"


def test_steps_add_up_to_the_rule():
    m = pd.DataFrame([{**GOOD, **c} for c, _ in CASES])
    keep = pd.Series(True, index=m.index)
    for _, mask in cr.steps(m):
        keep &= mask
    assert keep.equals(cr.clean_mask(m))
