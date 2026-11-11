"""The movie graph: movies, the people who made them, and their studios and
distributors, as one network to load and analyse.

Run from the data folder after graph_tables.sql, resolve_companies.py and
wiki_producers.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/build_graph.py

Nodes (features/graph/nodes.csv), one row each, keyed by "id":
- movie: IMDb ID (tt...), with title, year, genres, votes, rating, budget and
  gross in USD where both are USD, multiple (gross / budget), Rotten Tomatoes,
  Metacritic, vote tier, and clean (1 when the movie passes the clean
  theatrical rule in scripts/imdb/genome_stats.py).
- person: IMDb ID (nm...), with name, birth and death year, and the roles they
  hold anywhere in the graph.
- company: a studio or distributor, "company:<Wikidata ID>" when
  resolve_companies.py gave its name an ID, else "company:name:<name>". Name
  variants share a node only when they share an ID, so "Miramax Films" and
  "Miramax" are one node, and a label such as "Magnet Releasing" stays apart
  from its parent. A company that both produced and distributed films is one
  node with both roles. Fields: qid, the names it appears under, and roles.

Edges (features/graph/edges.csv) always run from a person or company to a
movie: role (actor, director, writer, producer, composer,
cinematographer, editor, studio, distributor), ordering (billing order, actors
only), year (the movie's) and origin: imdb for IMDb's credits, wikipedia for
companies and for producers the infobox names that IMDb's principals file
lacks (wiki_producers.py). Someone who both wrote and directed a film has
two edges to it.

features/graph/movie_graph.graphml holds the same graph for networkx or igraph.

Coverage: directors and writers are complete (IMDb's crew file). Actors,
producers and the other crew come from IMDb's principals file, which lists
about ten people a title, so a film's crew is often partial; producers are
topped up from Wikipedia.

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
COMPANIES = "features/graph/companies.csv"
PEOPLE = "features/graph/people.csv"
WIKI_PRODUCERS = "features/graph/wiki_producers.csv"
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
    film_company = defaultdict(set)  # (tconst, role) -> company node keys
    variants, croles = defaultdict(set), defaultdict(set)
    with open(COMPANIES, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            key = f"company:{r['qid']}" if r["qid"] else f"company:name:{r['name']}"
            film_company[(r["tconst"], r["role"])].add(key)
            variants[key].add(r["name"])
            croles[key].add(r["role"])
            if key not in nodes or r["label"] < nodes[key]["label"]:
                nodes[key] = {"type": "company", "label": r["label"], "qid": r["qid"] or None}
    for key in variants:
        nodes[key]["names"] = "|".join(sorted(variants[key]))
        nodes[key]["roles"] = "|".join(sorted(croles[key]))
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
        for kind in ("studio", "distributor"):
            for key in sorted(film_company[(r["tconst"], kind)]):  # two names of one company give one edge
                edges.append((key, r["tconst"], kind, None, year[r["tconst"]], "wikipedia"))

    roles = defaultdict(set)
    with open(CREDITS, newline="") as f:
        for r in csv.DictReader(f):
            roles[r["nconst"]].add(r["role"])
            edges.append((r["nconst"], r["tconst"], r["role"], num(r["ordering"], int), year[r["tconst"]], "imdb"))
    # Producers named in the Wikipedia infobox and matched to an IMDb ID (wiki_producers.py) that
    # IMDb's principals file doesn't list for the film.
    have = {(e[0], e[1]) for e in edges if e[2] == "producer"}
    wiki_names = {}
    with open(WIKI_PRODUCERS, newline="", encoding="utf8") as f:
        for r in csv.DictReader(f):
            if (r["nconst"], r["tconst"]) not in have and r["tconst"] in year:
                have.add((r["nconst"], r["tconst"]))
                roles[r["nconst"]].add("producer")
                wiki_names[r["nconst"]] = r["name"]
                edges.append((r["nconst"], r["tconst"], "producer", None, year[r["tconst"]], "wikipedia"))
    with open(PEOPLE, newline="") as f:
        for r in csv.DictReader(f):
            nodes[r["nconst"]] = {"type": "person", "label": r["name"], "birth_year": num(r["birth_year"], int),
                                  "death_year": num(r["death_year"], int),
                                  "roles": "|".join(sorted(roles[r["nconst"]]))}
    missing = {e[0] for e in edges} - set(nodes)
    for n in sorted(missing):  # absent from people.csv: Wikipedia-only producers, or missing from name.basics
        nodes[n] = {"type": "person", "label": wiki_names.get(n, n), "roles": "|".join(sorted(roles[n]))}

    cols = ["type", "label", "year", "genres", "votes", "rating", "budget_usd", "gross_usd", "multiple", "rt",
            "mc", "tier", "clean", "birth_year", "death_year", "roles", "qid", "names"]
    with open(NODES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"] + cols)
        for k in sorted(nodes):
            w.writerow([k] + ["" if nodes[k].get(c) is None else nodes[k].get(c) for c in cols])
    edges.sort(key=lambda e: (e[1], e[2], e[3] if e[3] is not None else math.inf, e[0]))
    with open(EDGES, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "role", "ordering", "year", "origin"])
        w.writerows(["" if v is None else v for v in e] for e in edges)

    g = nx.MultiDiGraph()
    for k in sorted(nodes):
        g.add_node(k, **{c: v for c, v in nodes[k].items() if v is not None})
    for s, t, role, order, y, origin in edges:
        g.add_edge(s, t, role=role, origin=origin, **({"ordering": order} if order is not None else {}),
                   **({"year": y} if y is not None else {}))
    nx.write_graphml(g, GRAPHML)
    by_type = defaultdict(int)
    for v in nodes.values():
        by_type[v["type"]] += 1
    print(f"wrote {NODES}, {EDGES}, {GRAPHML}: {dict(sorted(by_type.items()))}, {len(edges):,} edges"
          + (f", {len(missing)} people without a people.csv row (Wikipedia-only producers or missing from name.basics)"
             if missing else ""))


if __name__ == "__main__":
    main()
