"""No-leakage tests for scripts/network/film_features.py.

The script promises every feature of a film released in year Y is built only
from films released before Y. These tests check that promise on a small
synthetic graph: change or add a film in year Y, and no feature of any film
from year Y or earlier may move. A later film's features must move, so the
test can't pass by checking nothing.

    .venv/bin/python -m pytest tests/test_no_leakage.py
"""

import csv
import importlib.util
import os
import random
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get("FILM_FEATURES_PATH", ROOT / "scripts/network/film_features.py"))

NODE_COLS = ["id", "type", "label", "year", "genres", "votes", "rating", "budget_usd", "gross_usd",
             "multiple", "rt", "mc", "tier", "clean", "birth_year", "death_year", "roles", "qid", "names"]
EDGE_COLS = ["source", "target", "role", "ordering", "year", "origin"]
YEARS = list(range(1948, 1956))      # straddles FIRST_YEAR = 1950, so both network branches run
PIVOT_YEAR = 1952


def load_script():
    spec = importlib.util.spec_from_file_location("film_features", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def synthetic_graph(seed=0):
    """Three films a year drawn from a small, overlapping crew pool, so people and
    companies recur across years and every feature has history to read."""
    rng = random.Random(seed)
    pools = {"studio": ["s1", "s2", "s3"], "distributor": ["d1", "d2"], "producer": ["p1", "p2", "p3"],
             "writer": ["w1", "w2", "w3"], "director": ["r1", "r2", "r3"]}
    actors = [f"a{i}" for i in range(10)]
    nodes, edges = {}, []
    for y in YEARS:
        for k in range(3):
            t = f"tt{y}{k}"
            budget = rng.choice([1e6, 5e6, 2e7, 8e7])
            mult = round(rng.uniform(0.3, 6.0), 3)
            nodes[t] = {"id": t, "type": "movie", "label": t, "year": str(y),
                        "genres": rng.choice(["Horror", "Drama", "Comedy,Horror", "Action"]),
                        "budget_usd": str(budget), "gross_usd": str(budget * mult), "multiple": str(mult),
                        "clean": "0" if rng.random() < 0.15 else "1"}
            for role, pool in pools.items():
                for e in rng.sample(pool, rng.choice([1, 2]) if role in ("studio", "producer") else 1):
                    edges.append({"source": e, "target": t, "role": role, "ordering": "", "year": str(y)})
            for order, a in enumerate(rng.sample(actors, 6), start=1):
                edges.append({"source": a, "target": t, "role": "actor", "ordering": str(order), "year": str(y)})
    return nodes, edges


def run_features(tmp_path, nodes, edges):
    """Write the graph, run film_features.main() in tmp_path, return {tconst: row}."""
    graph = tmp_path / "features" / "graph"
    graph.mkdir(parents=True, exist_ok=True)
    with open(graph / "nodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=NODE_COLS)
        w.writeheader()
        for n in nodes.values():
            w.writerow({c: n.get(c, "") for c in NODE_COLS})
        for e in sorted({e["source"] for e in edges}):  # people and company nodes, as the real file has
            w.writerow({"id": e, "type": "person", "label": e})
    with open(graph / "edges.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EDGE_COLS)
        w.writeheader()
        for e in edges:
            w.writerow({c: e.get(c, "") for c in EDGE_COLS})
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        load_script().main()
    finally:
        os.chdir(cwd)
    with open(tmp_path / "features" / "film_network_features.csv", newline="") as f:
        return {r["tconst"]: r for r in csv.DictReader(f)}


def assert_no_leak(before, after, year, skip=()):
    """Rows for films from `year` or earlier must be identical; some later row must differ."""
    for t, row in before.items():
        if t in skip or int(row["year"]) > year:
            continue
        diff = {c: (row[c], after[t][c]) for c in row if row[c] != after[t][c]}
        assert not diff, f"{t} ({row['year']}) changed after editing a {year} film: {diff}"
    later = [t for t, r in before.items() if int(r["year"]) > year and t not in skip]
    assert any(before[t] != after[t] for t in later), "no later film changed, so the test checked nothing"


@pytest.mark.parametrize("film", [f"tt{PIVOT_YEAR}{k}" for k in range(3)])
def test_changing_a_films_result_does_not_leak_into_same_year_or_earlier(tmp_path, film):
    nodes, edges = synthetic_graph()
    before = run_features(tmp_path / "before", nodes, edges)
    n = nodes[film]
    n["multiple"] = str(float(n["multiple"]) * 7.3)
    n["budget_usd"] = str(float(n["budget_usd"]) * 3.1)
    n["genres"] = "Drama" if "Horror" in n["genres"] else "Horror"
    n["clean"] = "1"
    after = run_features(tmp_path / "after", nodes, edges)
    assert_no_leak(before, after, PIVOT_YEAR)


def test_films_with_no_history_get_no_history_features(tmp_path):
    """Films in the first year have nothing earlier to draw on. A leak that lets a film
    count itself (e.g. `<= y` instead of `< y`) shows up here as a non-zero value, even
    when it is constant and so invisible to the perturbation tests."""
    nodes, edges = synthetic_graph()
    rows = run_features(tmp_path, nodes, edges)
    first = [r for r in rows.values() if int(r["year"]) == YEARS[0]]
    assert first
    for r in first:
        for g in ["studio", "distributor", "producer", "writer", "director", "cast"]:
            assert r[f"{g}_track_n"] == "0", (r["tconst"], g)
            assert r[f"{g}_track"] == "", (r["tconst"], g)
            assert float(r[f"{g}_experience"]) == 0, (r["tconst"], g)
        assert float(r["cast_reteamed"]) == 0, r["tconst"]
        assert r["studio_prior_budget"] == "" and r["studio_prior_horror"] == "", r["tconst"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adding_a_film_does_not_leak_into_same_year_or_earlier(tmp_path, seed):
    nodes, edges = synthetic_graph(seed)
    before = run_features(tmp_path / "before", nodes, edges)
    t = f"tt{PIVOT_YEAR}9"
    nodes[t] = {"id": t, "type": "movie", "label": t, "year": str(PIVOT_YEAR), "genres": "Horror",
                "budget_usd": "1000000.0", "gross_usd": "9000000.0", "multiple": "9.0", "clean": "1"}
    crew = [("s1", "studio", ""), ("s2", "studio", ""), ("d1", "distributor", ""), ("p1", "producer", ""),
            ("w1", "writer", ""), ("r1", "director", "")] + [(f"a{i}", "actor", str(i + 1)) for i in range(6)]
    edges += [{"source": e, "target": t, "role": role, "ordering": o, "year": str(PIVOT_YEAR)} for e, role, o in crew]
    after = run_features(tmp_path / "after", nodes, edges)
    assert_no_leak(before, after, PIVOT_YEAR, skip={t})
