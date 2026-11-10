#!/usr/bin/env python3
"""Build a weighted co-appearance network from IMDb's public TSV datasets.

The default network uses non-adult movies with at least 10,000 IMDb votes.
For each movie, the first three distinct principal actors/actresses by IMDb's
ordering are retained.  An undirected edge counts the movies shared by a pair.

The script uses only the Python standard library and writes:
  nodes.csv, edges.csv, and actor_network.svg
"""

import argparse
import csv
import gzip
import html
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "raw"
DEFAULT_OUTPUT = ROOT / "build" / "actor-network"


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        yield from csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)


def load_movies(data_dir, min_votes):
    ratings = {}
    for row in rows(data_dir / "raw/title.ratings.tsv.gz"):
        if int(row["numVotes"]) >= min_votes:
            ratings[row["tconst"]] = row

    movies = {}
    for row in rows(data_dir / "raw/title.basics.tsv.gz"):
        if (
            row["titleType"] == "movie"
            and row["isAdult"] == "0"
            and row["tconst"] in ratings
        ):
            movies[row["tconst"]] = (
                row["primaryTitle"],
                ratings[row["tconst"]]["numVotes"],
            )
    return movies


def load_cast(data_dir, movies):
    cast = defaultdict(list)
    for row in rows(data_dir / "raw/title.principals.tsv.gz"):
        if row["tconst"] not in movies:
            continue
        if row["category"] not in {"actor", "actress"}:
            continue
        cast[row["tconst"]].append((int(row["ordering"]), row["nconst"]))

    selected = {}
    for title_id, credits in cast.items():
        actors = []
        seen = set()
        for _, name_id in sorted(credits):
            if name_id not in seen:
                actors.append(name_id)
                seen.add(name_id)
            if len(actors) == 3:
                break
        if len(actors) >= 2:
            selected[title_id] = actors
    return selected


def load_names(data_dir, actor_ids):
    names = {}
    for row in rows(data_dir / "raw/name.basics.tsv.gz"):
        if row["nconst"] in actor_ids:
            names[row["nconst"]] = row["primaryName"]
    return names


def build_network(movies, cast):
    edge_weights = Counter()
    movie_counts = Counter()
    for title_id, actors in cast.items():
        movie_counts.update(actors)
        edge_weights.update(itertools.combinations(sorted(actors), 2))
    return movie_counts, edge_weights


def write_tables(output, movies, cast, names, movie_counts, edge_weights):
    output.mkdir(parents=True, exist_ok=True)
    node_ids = sorted(movie_counts, key=lambda x: (-movie_counts[x], names.get(x, x)))
    with (output / "nodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actor_id", "actor_name", "movie_count", "weighted_degree"])
        weighted_degree = Counter()
        for (a, b), weight in edge_weights.items():
            weighted_degree[a] += weight
            weighted_degree[b] += weight
        for actor_id in node_ids:
            writer.writerow(
                [actor_id, names.get(actor_id, actor_id), movie_counts[actor_id], weighted_degree[actor_id]]
            )

    with (output / "edges.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actor_id_a", "actor_id_b", "weight", "shared_movie_count"])
        for (a, b), weight in sorted(edge_weights.items()):
            writer.writerow([a, b, weight, weight])

    with (output / "movies_used.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["title_id", "title", "votes", "actor_ids"])
        for title_id in sorted(cast):
            title, votes = movies[title_id]
            writer.writerow([title_id, title, votes, "|".join(cast[title_id])])


def write_svg(output, names, movie_counts, edge_weights, plot_nodes):
    selected = {
        actor_id
        for actor_id, _ in sorted(
            movie_counts.items(), key=lambda item: (-item[1], names.get(item[0], item[0]))
        )[:plot_nodes]
    }
    edges = [(a, b, w) for (a, b), w in edge_weights.items() if a in selected and b in selected]
    ranked = sorted(selected, key=lambda x: (-movie_counts[x], names.get(x, x)))
    width, height = 1400, 1000
    center_x, center_y = width / 2, height / 2
    radius = min(width, height) * 0.39
    positions = {}
    for index, actor_id in enumerate(ranked):
        angle = 2 * math.pi * index / max(1, len(ranked)) - math.pi / 2
        positions[actor_id] = (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))
    max_weight = max((weight for _, _, weight in edges), default=1)
    max_movies = max((movie_counts[actor_id] for actor_id in selected), default=1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>IMDb principal-actor co-appearance network</title>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="40" font-family="sans-serif" font-size="24">IMDb actor co-appearance network</text>',
        f'<text x="30" y="67" font-family="sans-serif" font-size="14">Top {len(selected)} actors by movie count; edge width = shared movies</text>',
    ]
    for a, b, weight in edges:
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        stroke_width = 0.7 + 4.5 * math.sqrt(weight / max_weight)
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="#9aa7b2" stroke-opacity="0.55" stroke-width="{stroke_width:.2f}"/>'
        )
    for actor_id in ranked:
        x, y = positions[actor_id]
        node_radius = 4 + 12 * math.sqrt(movie_counts[actor_id] / max_movies)
        label = html.escape(names.get(actor_id, actor_id))
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_radius:.1f}" fill="#276fbf" opacity="0.9"/>')
        if movie_counts[actor_id] >= max_movies * 0.35:
            lines.append(
                f'<text x="{x + node_radius + 4:.1f}" y="{y + 4:.1f}" font-family="sans-serif" '
                f'font-size="12">{label}</text>'
            )
    lines.append("</svg>")
    (output / "actor_network.svg").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--min-votes",
        type=int,
        default=10_000,
        help="minimum IMDb votes per movie; use 0 for all non-adult movies",
    )
    parser.add_argument(
        "--plot-nodes",
        type=int,
        default=150,
        help="number of highest-movie-count actors included in the SVG",
    )
    args = parser.parse_args()

    movies = load_movies(args.data_dir, args.min_votes)
    cast = load_cast(args.data_dir, movies)
    names = load_names(args.data_dir, {actor for actors in cast.values() for actor in actors})
    movie_counts, edge_weights = build_network(movies, cast)
    write_tables(args.output, movies, cast, names, movie_counts, edge_weights)
    write_svg(args.output, names, movie_counts, edge_weights, args.plot_nodes)
    print(f"movies selected: {len(movies):,}")
    print(f"movies with at least two principal actors: {len(cast):,}")
    print(f"actors: {len(movie_counts):,}")
    print(f"edges: {len(edge_weights):,}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
