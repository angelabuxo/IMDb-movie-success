"""Producers from the Wikipedia infobox, matched to IMDb person IDs, to fill
in the producers IMDb's principals file leaves out (it lists about ten people
a film).

Run from the data folder after graph_tables.sql, before build_graph.py:

    cd data/raw && ../../.venv/bin/python ../../scripts/network/wiki_producers.py

A Wikipedia producer name (an article title such as "Tony Adams (producer)",
with the bracket dropped) is matched to name.basics by exact name, ignoring
case. It gets an IMDb ID only when the match is unambiguous:

1. credited: exactly one person with that name is credited on the same film
   in IMDb, in any role;
2. unique producer: otherwise, exactly one person with that name lists
   producer among their professions.

Anything else is left out. A third rule, the only person in IMDb with that
name, was tried and dropped: only 6% of its 134 matches had any other credit
in the graph, against 72% within 15 years for rule 2, so it mostly found
namesakes. The check printed at the end is the share of
matched names IMDb itself credits as producer on the same film, where IMDb
lists that film's producers.

Writes features/graph/wiki_producers.csv: tconst, nconst, name, method.
"""

import csv
import gzip
import re
from collections import defaultdict

INFOBOX = ["features/wiki_infobox.csv", "features/wiki_infobox_10k.csv"]
CREDITS = "features/graph/credits.csv"
NAMES = "raw/name.basics.tsv.gz"
OUT = "features/graph/wiki_producers.csv"


def clean(n):
    return re.sub(r"\s*\([^)]*\)\s*$", "", n).strip()


def main():
    wanted = defaultdict(set)  # tconst -> producer names
    for path in INFOBOX:
        with open(path, newline="", encoding="utf8") as f:
            for r in csv.DictReader(f):
                for n in (r.get("producers") or "").split("|"):
                    n = clean(n)
                    if n and len(n) > 3 and not re.search(r"\d", n):
                        wanted[r["tconst"]].add(n)
    keys = {n.casefold() for ns in wanted.values() for n in ns}

    people = defaultdict(list)  # casefolded name -> [(nconst, is_producer)]
    with gzip.open(NAMES, "rt", encoding="utf8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[1].casefold() in keys:
                people[p[1].casefold()].append((p[0], "producer" in p[4].split(",")))

    credited, imdb_producers = defaultdict(set), defaultdict(set)
    with open(CREDITS, newline="") as f:
        for r in csv.DictReader(f):
            credited[r["tconst"]].add(r["nconst"])
            if r["role"] == "producer":
                imdb_producers[r["tconst"]].add(r["nconst"])

    rows, how, agree, checked = [], defaultdict(int), 0, 0
    for t in sorted(wanted):
        for n in sorted(wanted[t]):
            cands = people.get(n.casefold(), [])
            on_film = [c for c, _ in cands if c in credited[t]]
            prod = [c for c, is_p in cands if is_p]
            if len(on_film) == 1:
                pick, method = on_film[0], "credited"
            elif not on_film and len(prod) == 1:
                pick, method = prod[0], "unique producer"
            else:
                how["ambiguous" if cands else "not in IMDb"] += 1
                continue
            how[method] += 1
            rows.append({"tconst": t, "nconst": pick, "name": n, "method": method})
            if imdb_producers[t]:
                checked += 1
                agree += pick in imdb_producers[t]
    with open(OUT, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=["tconst", "nconst", "name", "method"])
        w.writeheader()
        w.writerows(rows)
    new = sum(1 for r in rows if r["nconst"] not in imdb_producers[r["tconst"]])
    print(f"{sum(len(v) for v in wanted.values()):,} Wikipedia producer credits on {len(wanted):,} films: "
          f"{dict(sorted(how.items()))}")
    print(f"matched {len(rows):,}; {new:,} are producer credits IMDb's principals file lacks")
    print(f"check: where IMDb lists the film's producers, {agree:,} of {checked:,} matches "
          f"({agree / max(checked, 1):.0%}) are among them")


if __name__ == "__main__":
    main()
