"""Who is in each people-network community: the profile table behind the paper's description of the communities.

Rebuilds the Louvain partition of network_communities.py (same links, same seeds, checked against its saved
modularity) and, for each community with at least MIN_SIZE films, in the same order as community_returns.csv:
years, median budget, median gross/budget multiple, and the studios, directors, producers and top-three actors
credited on most of its films (name and number of films). Fast: no null, no shuffles.

What this measures: which people and companies hold each community together. The names are the most frequent
credits, not every film; a single shared studio or producer can pull an unrelated film into a community.
Outputs: analysis/compare/outputs/communities/community_profiles.csv (names and numbers only).
Run:  IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.compare.network_community_profiles
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from analysis.compare import network_communities as nc
from analysis.compare.network_by_genre import members
from analysis.genre.q1_genre import DATA, load, prepare

TOP = 3


def main():
    films, genres = load()
    f, keep, M = prepare(films, genres, "in_common_set")
    G = nc.weighted_links(f)
    nodes = np.array(sorted(G.nodes()))
    parts, mods = nc.louvain(G, nc.SEEDS)
    best = int(np.argmax(mods))
    saved = json.loads((nc.OUT / "summary.json").read_text())
    assert round(mods[best], 4) == saved["modularity"], "partition differs from network_communities.py's saved run"
    lab = nc.labels(parts[best], list(nodes))
    sizes = Counter(lab)
    big = [c for c, n in sizes.most_common() if n >= nc.MIN_SIZE]
    ret = pd.read_csv(nc.OUT / "community_returns.csv")
    assert len(ret) == len(big)
    fi = f.set_index("imdb_id")
    mem = members(set(f["imdb_id"]))
    name = pd.read_csv(DATA / "features" / "graph" / "nodes.csv", usecols=["id", "label"]).set_index("id")["label"]
    rows = []
    for j, c in enumerate(big):
        S = set(nodes[lab == c])
        sub = fi.loc[sorted(S)]
        row = {"community": j + 1, "label": ret.loc[j, "category"], "films": len(S),
               "year_q25": int(sub.year.quantile(.25)), "year_median": int(sub.year.median()), "year_q75": int(sub.year.quantile(.75)),
               "median_budget_musd": round(float(sub.budget.median()) / 1e6, 1),
               "median_multiple": round(float(np.exp(sub.ln_gross_over_budget.median())), 2),
               "excess": ret.loc[j, "excess"], "p_holm": ret.loc[j, "p_holm"]}
        for role in ("studio", "director", "producer", "actor"):
            cnt = Counter({e: len(fs & S) for e, fs in mem[role].items() if len(fs & S) >= 2})
            row[f"top_{role}s"] = "; ".join(f"{name.get(e, e)} ({k})" for e, k in cnt.most_common(TOP))
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(nc.OUT / "community_profiles.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(d.drop(columns=["label"]).to_string(index=False))


if __name__ == "__main__":
    main()
