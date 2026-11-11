"""Compute movie ROI and add actor ROI summaries to the network page.

Run from the repository root after extracting the data bundles:

    python scripts/network/actor_roi.py

Movie ROI is (USD gross - USD budget) / USD budget.  Movies without both
positive USD values are excluded from the ROI summaries.
"""

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import networkx as nx

DEFAULT_INFOBOX = Path("data/raw/features/wiki_infobox_10k.csv")
DEFAULT_EDGES = Path("data/raw/features/graph/edges.csv")
DEFAULT_NODES = Path("data/raw/features/graph/nodes.csv")
DEFAULT_PAGE = Path("build/actor-network/actor_network_interactive.html")


def read_movie_roi(path: Path) -> dict[str, dict[str, object]]:
    movies = {}
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["budget_cur"] != "USD" or row["gross_cur"] != "USD":
                continue
            try:
                budget = float(row["budget"])
                gross = float(row["gross"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(budget) or not math.isfinite(gross) or budget <= 0:
                continue
            movies[row["tconst"]] = {
                "tconst": row["tconst"],
                "title": row["title"],
                "budget_usd": budget,
                "gross_usd": gross,
                "roi": (gross - budget) / budget,
            }
    return movies


def read_actor_movies(path: Path) -> dict[str, set[str]]:
    actor_movies = defaultdict(set)
    with path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["role"] == "actor":
                actor_movies[row["source"]].add(row["target"])
    return actor_movies


def write_movie_output(path: Path, actor_movies: dict[str, set[str]],
                       movies: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    movie_ids = sorted({movie_id for ids in actor_movies.values() for movie_id in ids if movie_id in movies})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["tconst", "title", "budget_usd", "gross_usd", "roi"])
        writer.writeheader()
        writer.writerows(movies[movie_id] for movie_id in movie_ids)


def actor_summaries(actor_movies: dict[str, set[str]],
                    movies: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    summaries = {}
    for actor_id, movie_ids in actor_movies.items():
        values = [float(movies[movie_id]["roi"]) for movie_id in movie_ids if movie_id in movies]
        summaries[actor_id] = {
            "roi_movie_count": len(values),
            "roi_total": sum(values) if values else None,
            "roi_median": median(values) if values else None,
            "roi_average": mean(values) if values else None,
        }
    return summaries


def write_actor_output(path: Path, summaries: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["actor_id", "roi_movie_count", "roi_total", "roi_median", "roi_average"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for actor_id in sorted(summaries):
            writer.writerow({"actor_id": actor_id, **summaries[actor_id]})


def build_networks(edges_path: Path, nodes_path: Path,
                   movies: dict[str, dict[str, object]],
                   summaries: dict[str, dict[str, object]]) -> dict[str, dict[str, list]]:
    names = {}
    movie_names = {}
    with nodes_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            names[row["id"]] = row["label"]
            if row["type"] == "movie":
                movie_names[row["id"]] = row["label"]
    movie_people = defaultdict(set)
    actor_movies = defaultdict(set)
    director_movies = defaultdict(set)
    with edges_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["role"] == "actor":
                actor_movies[row["source"]].add(row["target"])
                movie_people[row["target"]].add(row["source"])
            elif row["role"] == "director":
                director_movies[row["source"]].add(row["target"])
                movie_people[row["target"]].add(row["source"])
    movie_directors = defaultdict(set)
    for director, movie_ids_for_director in director_movies.items():
        for movie_id in movie_ids_for_director:
            movie_directors[movie_id].add(director)

    def network(groups, entity_movies, limit=150, preset_weights=None, selected_entities=None):
        weights = defaultdict(int)
        if preset_weights is not None:
            weights.update(preset_weights)
        else:
            for members in groups.values():
                for first, second in combinations(sorted(members), 2):
                    weights[(first, second)] += 1
        degrees = defaultdict(int)
        for (first, second), weight in weights.items():
            degrees[first] += weight
            degrees[second] += weight
        selected = set(selected_entities or sorted(entity_movies, key=lambda x: (-len(entity_movies[x]), x))[:limit])
        selected_edges = [(a, b, weight) for (a, b), weight in weights.items()
                          if a in selected and b in selected]
        graph = nx.Graph()
        graph.add_nodes_from(selected)
        graph.add_weighted_edges_from(selected_edges)
        closeness = nx.closeness_centrality(graph, distance=lambda _, __, data: 1 / data["weight"])
        betweenness = nx.betweenness_centrality(graph, weight="weight", normalized=True)
        eigenvector = nx.eigenvector_centrality(graph, weight="weight", max_iter=500) if graph.number_of_edges() else {}
        nodes = []
        for index, entity in enumerate(sorted(selected, key=lambda x: (-len(entity_movies[x]), x))):
            values = [float(movies[movie_id]["roi"]) for movie_id in entity_movies[entity] if movie_id in movies]
            degree = graph.degree(entity)
            nodes.append({
                "id": entity,
                "name": names.get(entity, movie_names.get(entity, entity)),
                "movieCount": len(entity_movies[entity]),
                "weightedDegree": sum(weight for _, _, weight in graph.edges(entity, data="weight")),
                "collaboratorCount": degree,
                "degree": degree,
                "closeness": round(closeness.get(entity, 0), 4),
                "betweenness": round(betweenness.get(entity, 0), 4),
                "eigenvector": round(eigenvector.get(entity, 0), 4),
                "clustering": round(nx.clustering(graph, entity, weight="weight"), 4),
                "x": 700 + (index % 15) * 16,
                "y": 110 + (index // 15) * 16,
                "radius": 8 + min(8, len(entity_movies[entity]) ** 0.5),
                "roi_movie_count": len(values),
                "roi_total": sum(values) if values else None,
                "roi_median": median(values) if values else None,
                "roi_average": mean(values) if values else None,
            })
        return {"nodes": nodes, "edges": [
            {"a": a, "b": b, "weight": weight, "width": 1.5 + 1.04 * math.sqrt(weight)}
            for a, b, weight in selected_edges
        ]}

    movie_ids = sorted(movie_people, key=lambda x: (-len(movie_people[x]), x))[:150]
    movie_entities = {movie_id: {movie_id} for movie_id in movie_ids}
    movie_groups = defaultdict(set)
    person_movies = defaultdict(set)
    for movie_id, people in movie_people.items():
        for person in people:
            person_movies[person].add(movie_id)
    for person, movie_ids_for_person in person_movies.items():
        for movie_id in movie_ids_for_person:
            if movie_id in movie_entities:
                movie_groups[person].add(movie_id)
    actor_edges_file = Path("build/actor-network/edges.csv")
    actor_nodes_file = Path("build/actor-network/nodes.csv")
    actor_weights = {}
    actor_selected = None
    if actor_edges_file.exists() and actor_nodes_file.exists():
        with actor_edges_file.open(newline="", encoding="utf-8-sig") as file:
            actor_weights = {
                (min(row["actor_id_a"], row["actor_id_b"]), max(row["actor_id_a"], row["actor_id_b"])): int(row["weight"])
                for row in csv.DictReader(file)
            }
        with actor_nodes_file.open(newline="", encoding="utf-8-sig") as file:
            actor_selected = [row["actor_id"] for row in csv.DictReader(file)][:150]
    return {
        "actor": network(movie_people, actor_movies, preset_weights=actor_weights,
                         selected_entities=actor_selected),
        "director": network(
            {actor: {director for movie_id in movie_ids_for_actor
                     for director in movie_directors[movie_id]}
             for actor, movie_ids_for_actor in actor_movies.items()},
            director_movies,
        ),
        "movie": network(movie_groups, movie_entities),
    }


def update_page(path: Path, summaries: dict[str, dict[str, object]],
                networks: dict[str, dict[str, list]]) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^const (?:networkData|graph) = (.*);$", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find graph JSON in {path}")
    if match.group(0).startswith("const graph"):
        graph = json.loads(match.group(1))
        for node in graph["nodes"]:
            node.update(summaries.get(node["id"], {}))
        networks["actor"] = graph
    graph_json = json.dumps(networks, separators=(",", ":"), ensure_ascii=True)
    replacement = f"const networkData = {graph_json};\nlet graph = networkData[new URLSearchParams(location.search).get(\"network\") || \"actor\"];"
    suffix = re.match(r"\nlet graph = .*;\s*", text[match.end():])
    end = match.end() + (suffix.end() if suffix else 0)
    text = text[:match.start()] + replacement + text[end:]
    selector = """const networkSelect = document.getElementById("networkType");
networkSelect.value = new URLSearchParams(location.search).get("network") || "actor";
networkSelect.addEventListener("change", event => {
  window.location.search = `?network=${encodeURIComponent(event.target.value)}`;
});
"""
    marker = 'const svg = document.getElementById("network");'
    if selector not in text:
        text = text.replace(marker, selector + marker, 1)
    marker = "function showActorDetails(node, pinned = false) {"
    formatter = """function formatRoi(value) {
  return value === null ? "Not available" : `${(value * 100).toFixed(1)}%`;
}

"""
    if formatter not in text:
        if marker not in text:
            raise ValueError(f"Could not find actor detail renderer in {path}")
        text = text.replace(marker, formatter + marker, 1)
    old = """      <dt>Movies in network</dt><dd>${node.movieCount.toLocaleString()}</dd>
      <dt>Collaborators</dt>"""
    old_generic = """      <dt>Movies represented</dt><dd>${node.movieCount.toLocaleString()}</dd>
      <dt>Collaborators</dt>"""
    new = """      <dt>Movies in network</dt><dd>${node.movieCount.toLocaleString()}</dd>
      <dt>Movies with ROI</dt><dd>${node.roi_movie_count.toLocaleString()}</dd>
      <dt>Total ROI</dt><dd>${formatRoi(node.roi_total)}</dd>
      <dt>Median ROI</dt><dd>${formatRoi(node.roi_median)}</dd>
      <dt>Average ROI</dt><dd>${formatRoi(node.roi_average)}</dd>
      <dt>Collaborators</dt>"""
    if old not in text and old_generic not in text and new not in text and "Movies with ROI" not in text:
        raise ValueError(f"Could not find actor detail fields in {path}")
    if old in text:
        text = text.replace(old, new, 1)
    elif old_generic in text:
        text = text.replace(old_generic, new.replace("Movies in network", "Movies represented"), 1)
    path.write_text(text, encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infobox", type=Path, default=DEFAULT_INFOBOX)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--page", type=Path, default=DEFAULT_PAGE)
    parser.add_argument("--output-dir", type=Path, default=Path("build/actor-network"))
    args = parser.parse_args()

    movies = read_movie_roi(args.infobox)
    actor_movies = read_actor_movies(args.edges)
    summaries = actor_summaries(actor_movies, movies)
    networks = build_networks(args.edges, args.nodes, movies, summaries)
    write_movie_output(args.output_dir / "movie_roi.csv", actor_movies, movies)
    write_actor_output(args.output_dir / "actor_roi.csv", summaries)
    update_page(args.page, summaries, networks)
    usable_appearances = sum(
        movie_id in movies for movie_ids in actor_movies.values() for movie_id in movie_ids
    )
    usable_movies = len({movie_id for movie_ids in actor_movies.values() for movie_id in movie_ids if movie_id in movies})
    print(f"wrote ROI for {usable_appearances:,} actor-movie appearances")
    print(f"wrote {usable_movies:,} movies with actor data")
    print(f"wrote {usable_movies:,} movie ROI rows and {len(summaries):,} actor ROI rows")
    print(f"updated {args.page}")


if __name__ == "__main__":
    main()
