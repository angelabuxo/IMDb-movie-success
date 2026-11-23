"""The final set for Q3 (decided universe: all films of movies_features_10k.csv).

Read from the shared table built by scripts/imdb/build_analysis_set.py (features/film_analysis_set.csv),
so there is one definition of the set. `in_common_set` is True for a film with a MovieSum script, the shared
money rule (analysis.clean_rule, no net_gross flag, no flag on the budget) and at least 500 tokens in both
the dialogue and the description layer. shared_rule_check.py rebuilds the same set independently as a cross-check.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from genre_counts import DATA  # noqa: E402

TABLE = DATA / "features" / "film_analysis_set.csv"


def read_table():
    if not TABLE.exists():
        raise SystemExit(f"{TABLE} is missing: run scripts/imdb/build_analysis_set.py from the data folder first")
    with open(TABLE, newline="") as f:
        return list(csv.DictReader(f))


def final_films(tier=None):
    """Films of the final set as dicts (id, genres, year, decade, ln, tier). tier='100k+' keeps the robustness subset."""
    out = []
    for r in read_table():
        if r["in_common_set"] != "True" or (tier and r["tier"] != tier):
            continue
        genres = [g for g in r["genres"].split(",") if g]
        year = int(r["year"])
        out.append({"id": r["imdb_id"], "genres": genres, "year": year, "decade": year // 10 * 10,
                    "ln": float(r["ln_gross_over_budget"]), "tier": r["tier"],
                    "name": r["title"]})
    return out


def tiny_usable_ids():
    """Films with a script and usable money whose dialogue or description layer is under the token rule."""
    return {r["imdb_id"] for r in read_table()
            if r["in_moviesum"] == "True" and r["usable_money"] == "True" and not r["in_common_set"] == "True"}
