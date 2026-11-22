"""Assortativity of returns in a film-only collaboration network.

Run from the data folder after ``build_graph.py``:

    ../../.venv/bin/python ../../scripts/network/movie_assortativity.py

The nodes are clean films with a usable gross and budget. Two films are linked
when they share a director, a top-billed actor, a producer, or a studio. The
outcome is ``ln(gross / budget)``. For each graph, the observed Pearson
assortativity of the outcome is compared with the same graph after returns
are permuted between films. This is a descriptive clustering test, not a
prediction or a causal estimate.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

NODES = "features/graph/nodes.csv"
EDGES = "features/graph/edges.csv"
SUMMARY = Path(__file__).resolve().parents[2] / "analysis" / "network" / "movie_assortativity.json"
ROLES = ("director", "main_actor", "producer", "studio")


def film_data(nodes_path=NODES):
    """Return clean-film data keyed by IMDb ID."""
    films = {}
    with open(nodes_path, newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if row["type"] != "movie" or row["clean"] != "1" or not row["multiple"]:
                continue
            films[row["id"]] = {
                "return": math.log(float(row["multiple"])),
                "budget": float(row["budget_usd"]),
                "year": int(row["year"]),
                "genres": [g for g in row["genres"].split(",") if g],
            }
    return films


def movie_members(edges_path=EDGES, film_ids=None):
    """Return ``(role, entity) -> films`` for the four requested link types."""
    film_ids = set(film_ids or ())
    members = defaultdict(set)
    with open(edges_path, newline="", encoding="utf8") as handle:
        for row in csv.DictReader(handle):
            if row["target"] not in film_ids:
                continue
            if row["role"] == "director":
                role = "director"
            elif row["role"] == "producer":
                role = "producer"
            elif row["role"] == "studio":
                role = "studio"
            elif row["role"] == "actor" and row["ordering"] and int(row["ordering"]) == 1:
                role = "main_actor"
            else:
                continue
            members[(role, row["source"])].add(row["target"])
    return members


def movie_graph(film_ids, members, role=None):
    """Build a simple undirected film graph, with shared roles on each edge."""
    graph = nx.Graph()
    graph.add_nodes_from(film_ids)
    for (member_role, _), films in members.items():
        if role is not None and member_role != role:
            continue
        for left, right in combinations(sorted(films), 2):
            if graph.has_edge(left, right):
                graph[left][right]["roles"].add(member_role)
            else:
                graph.add_edge(left, right, roles={member_role})
    return graph


def assortativity(graph, values):
    """Pearson correlation of outcome values at the two ends of each edge."""
    pairs = [(values[left], values[right]) for left, right in graph.edges]
    if len(pairs) < 2:
        return float("nan")
    left, right = np.asarray(pairs, dtype=float).T
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _edge_correlation(values, left, right):
    """Correlation for pre-indexed edge endpoints."""
    first, second = values[left], values[right]
    if np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def adjusted_returns(ids, films):
    """Residualise return against budget, release decade, and genre."""
    genres = sorted({genre for film in ids for genre in films[film]["genres"]})
    decades = sorted({films[film]["year"] // 10 for film in ids})[1:]
    x = np.array([
        [1.0, math.log(films[film]["budget"])]
        + [float(films[film]["year"] // 10 == decade) for decade in decades]
        + [float(genre in films[film]["genres"]) for genre in genres]
        for film in ids
    ])
    y = np.array([films[film]["return"] for film in ids])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def shuffle_results(graph, values, shuffles, rng):
    """Observed assortativity and a return-permutation null distribution."""
    nodes = list(graph.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    edge_index = np.asarray([(index[a], index[b]) for a, b in graph.edges], dtype=int).reshape(-1, 2)
    left, right = edge_index.T
    ordered = np.array([values[node] for node in nodes])
    observed = _edge_correlation(ordered, left, right)
    null = np.array([_edge_correlation(rng.permutation(ordered), left, right) for _ in range(shuffles)])
    return {
        "edges": graph.number_of_edges(),
        "observed": round(observed, 4),
        "null_mean": round(float(np.nanmean(null)), 4),
        "null_95": round(float(np.nanquantile(null, 0.95)), 4),
        "p_one_sided": round(float((1 + np.sum(null >= observed)) / (1 + shuffles)), 4),
    }


def run(shuffles=1000, seed=2805, nodes_path=NODES, edges_path=EDGES):
    films = film_data(nodes_path)
    ids = sorted(films)
    members = movie_members(edges_path, ids)
    rng = np.random.default_rng(seed)
    outcomes = {
        "ln_gross_over_budget": {film: films[film]["return"] for film in ids},
        "adjusted": dict(zip(ids, adjusted_returns(ids, films))),
    }
    results = {}
    for graph_name, role in [("combined", None), *[(r, r) for r in ROLES]]:
        graph = movie_graph(ids, members, role)
        results[graph_name] = {
            measure: shuffle_results(
                graph, values, shuffles, rng
            )
            for measure, values in outcomes.items()
        }
        results[graph_name]["nodes"] = graph.number_of_nodes()
        results[graph_name]["isolated_nodes"] = sum(degree == 0 for _, degree in graph.degree)
    return {"films": len(ids), "shuffles": shuffles, "seed": seed, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2805)
    args = parser.parse_args()
    summary = run(args.shuffles, args.seed)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    for graph_name, result in summary["results"].items():
        raw = result["ln_gross_over_budget"]
        print(f"{graph_name:12s} edges {raw['edges']:8d} "
              f"assortativity {raw['observed']:+.3f} "
              f"(shuffle 95% {raw['null_95']:+.3f}, p {raw['p_one_sided']:.4f})")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
