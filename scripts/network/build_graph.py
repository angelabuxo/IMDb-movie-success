"""The movie graph: movies, the people who made them, and their studios and
distributors, as one network to load and analyse.

Run from the data folder after graph_tables.sql:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/build_graph.py

Nodes (features/graph/nodes.csv), one row each, keyed by "id":
- movie: IMDb ID (tt...), with title, year, genres, votes, rating, budget and
  gross in USD where both are USD, multiple (gross / budget), Rotten Tomatoes,
  Metacritic, vote tier, and clean (1 when the movie passes the clean
  theatrical rule in scripts/imdb/genome_stats.py).
- person: IMDb ID (nm...), with name, birth and death year, and the roles they
  hold anywhere in the graph.
- studio and distributor: "studio:<name>" or "distributor:<name>", from the
  Wikipedia infobox strings. Names aren't merged, so "Warner Bros." and
  "Warner Bros. Pictures" are two nodes.

Edges (features/graph/edges.csv) always run from a person, studio or
distributor to a movie: role (actor, director, writer, producer, composer,
cinematographer, editor, studio, distributor), ordering (billing order, actors
only) and year (the movie's). Someone who both wrote and directed a film has
two edges to it.

features/graph/movie_graph.graphml holds the same graph for networkx or igraph.

Coverage: directors and writers are complete (IMDb's crew file). Actors,
producers and the other crew come from IMDb's principals file, which lists
about ten people a title, so a film's producers and crew are often partial.

Leakage: success makes people more central, so for anything that predicts a
movie's result from the network, build it from edges whose year is before
that movie's year.
"""

import csv
import math
import re
from collections import defaultdict

import networkx as nx

MOVIES = "features/movies_features_10k.csv"
CREDITS = "features/graph/credits.csv"
PEOPLE = "features/graph/people.csv"
NODES = "features/graph/nodes.csv"
EDGES = "features/graph/edges.csv"
GRAPHML = "features/graph/movie_graph.graphml"


def num(x, cast=float):
    try:
        return cast(x) if x not in (None, "") else None
    except ValueError:
        return None


def clean(r):
    """The clean theatrical rule from scripts/imdb/genome_stats.py."""
    return (r["budget_cur"] == "USD" and r["gross_cur"] == "USD" and float(r["b"] or 0) >= 1000
            and float(r["g"] or 0) >= 1000 and "rentals" not in r["gross_flag"]
            and "regional" not in r["gross_flag"]
            and not re.search("netflix|amazon|apple|hulu", r["distributors"] or "", re.I)
            and int(r["y"] or 0) < 2026)


def main():
    nodes, edges = {}, []
    with open(MOVIES, newline="") as f:
        movies = list(csv.DictReader(f))
    year = {}
    for r in movies:
        usd = r["budget_cur"] == "USD" and r["gross_cur"] == "USD"
        b, g = (num(r["b"]), num(r["g"])) if usd else (None, None)
        year[r["tconst"]] = num(r["y"], int)
        nodes[r["tconst"]] = {
            "type": "movie", "label": r["primaryTitle"], "year": year[r["tconst"]], "genres": r["genres"],
            "votes": num(r["votes"], int), "rating": num(r["rating"]), "budget_usd": b, "gross_usd": g,
            "multiple": round(g / b, 4) if b and g else None, "rt": num(r["rt"], int), "mc": num(r["mc"], int),
            "tier": r["tier"], "clean": int(clean(r)),
        }
        for kind, field in (("studio", "studios"), ("distributor", "distributors")):
            for name in sorted({s.strip() for s in (r[field] or "").split("|") if s.strip()}):
                key = f"{kind}:{name}"
                nodes.setdefault(key, {"type": kind, "label": name})
                edges.append((key, r["tconst"], kind, None, year[r["tconst"]]))

    roles = defaultdict(set)
    with open(CREDITS, newline="") as f:
        for r in csv.DictReader(f):
            roles[r["nconst"]].add(r["role"])
            edges.append((r["nconst"], r["tconst"], r["role"], num(r["ordering"], int), year[r["tconst"]]))
    with open(PEOPLE, newline="") as f:
        for r in csv.DictReader(f):
            nodes[r["nconst"]] = {"type": "person", "label": r["name"], "birth_year": num(r["birth_year"], int),
                                  "death_year": num(r["death_year"], int),
                                  "roles": "|".join(sorted(roles[r["nconst"]]))}
    missing = {e[0] for e in edges} - set(nodes)
    for n in sorted(missing):  # credited in IMDb but absent from name.basics
        nodes[n] = {"type": "person", "label": n, "roles": "|".join(sorted(roles[n]))}

    cols = ["type", "label", "year", "genres", "votes", "rating", "budget_usd", "gross_usd", "multiple", "rt",
            "mc", "tier", "clean", "birth_year", "death_year", "roles"]
    with open(NODES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + cols)
        for k in sorted(nodes):
            w.writerow([k] + ["" if nodes[k].get(c) is None else nodes[k].get(c) for c in cols])
    edges.sort(key=lambda e: (e[1], e[2], e[3] if e[3] is not None else math.inf, e[0]))
    with open(EDGES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "role", "ordering", "year"])
        w.writerows(["" if v is None else v for v in e] for e in edges)

    g = nx.MultiDiGraph()
    for k in sorted(nodes):
        g.add_node(k, **{c: v for c, v in nodes[k].items() if v is not None})
    for s, t, role, order, y in edges:
        g.add_edge(s, t, role=role, **({"ordering": order} if order is not None else {}),
                   **({"year": y} if y is not None else {}))
    nx.write_graphml(g, GRAPHML)
    by_type = defaultdict(int)
    for v in nodes.values():
        by_type[v["type"]] += 1
    print(f"wrote {NODES}, {EDGES}, {GRAPHML}: {dict(sorted(by_type.items()))}, {len(edges):,} edges"
          + (f", {len(missing)} people missing from name.basics" if missing else ""))


if __name__ == "__main__":
    main()
