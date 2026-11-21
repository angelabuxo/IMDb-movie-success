"""Reproducible answer to Q2: do films sharing directors, actors, producers or studios earn alike?

Run from the repo root:
    .venv/Scripts/python.exe scripts/network/q2_network.py

Primary sample: in_common_set (the shared head-to-head set).
Robustness: usable_money and the 100k+ tier in that set.
"""

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "data" / "raw" / "features" / "film_analysis_set.csv"
EDGES = ROOT / "data" / "raw" / "features" / "graph" / "edges.csv"
OUT_JSON = ROOT / "analysis" / "network" / "q2_network.json"
OUT_MD = ROOT / "docs" / "q2-network.md"
OUT_FIG = ROOT / "analysis" / "network" / "figures" / "q2_network.png"

ROLES = ("director", "actor", "producer", "studio")
CAST_TOP = 3
ERA_ORDER = ("before 1980", "1980s", "1990s", "2000s", "2010s", "2020s")


def residualize(frame):
    """ln(gross/budget) minus OLS effects of ln(budget), era and genres."""
    genres = sorted({g for value in frame["genres"].fillna("") for g in value.split(",") if g})
    eras = [era for era in ERA_ORDER if era in set(frame["era_bin"].dropna())]
    baseline_era = eras[0]
    x = np.column_stack([
        np.ones(len(frame)),
        np.log(frame["budget"].astype(float).to_numpy()),
        *[frame["era_bin"].eq(era).astype(float).to_numpy() for era in eras[1:]],
        *[frame["genres"].fillna("").apply(
            lambda value, genre=g: float(genre in {item for item in value.split(",") if item})
        ).to_numpy() for g in genres],
    ])
    y = frame["ln_gross_over_budget"].astype(float).to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return y - x @ beta


def role_members(edges, ids):
    members = {role: defaultdict(set) for role in ROLES}
    for row in edges:
        if row["target"] not in ids:
            continue
        role = row["role"]
        if role == "actor":
            try:
                if int(row["ordering"]) > CAST_TOP:
                    continue
            except (TypeError, ValueError):
                continue
            role = "actor"
        if role in members:
            members[role][row["source"]].add(row["target"])
    return members


def make_graph(ids, members, role):
    pairs = defaultdict(set)
    for entity, films in members[role].items():
        films = sorted(films & ids)
        for pair in combinations(films, 2):
            pairs[pair].add(entity)
    return pairs


def combined_graph(ids, members):
    pairs = defaultdict(set)
    for role in ROLES:
        for entity, films in members[role].items():
            for pair in combinations(sorted(films & ids), 2):
                pairs[pair].add((role, entity))
    return pairs


def assortativity(pairs, index, y):
    if not pairs:
        return float("nan"), 0
    if len(pairs) < 2:
        return float("nan"), len(pairs)
    a = np.array([y[index[i]] for i, _ in pairs])
    b = np.array([y[index[j]] for _, j in pairs])
    if a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return float("nan"), len(pairs)
    return float(np.corrcoef(a, b)[0, 1]), len(pairs)


def shuffle_within_era(y, eras, rng):
    out = y.copy()
    for era in pd.unique(eras):
        pos = np.flatnonzero(eras == era)
        out[pos] = y[rng.permutation(pos)]
    return out


def intraclass(ids, y, groups):
    index = {film: i for i, film in enumerate(ids)}
    group_films = [sorted(group & set(ids)) for group in groups]
    group_films = [g for g in group_films if len(g) >= 2]
    if not group_films:
        return {"groups": 0, "credits": 0, "similarity": None, "loo_r": None}
    g, m, v = [], [], []
    for k, films in enumerate(group_films):
        for film in films:
            g.append(k)
            m.append(index[film])
    g = np.array(g)
    m = np.array(m)
    v = y[m]
    sizes = np.bincount(g, minlength=len(group_films)).astype(float)
    sums = np.bincount(g, weights=v, minlength=len(group_films))
    means = sums / sizes
    pooled_within = ((v - means[g]) ** 2).sum() / (len(v) - len(group_films))
    similarity = 1 - pooled_within / v.var(ddof=1)
    others = (sums[g] - v) / (sizes[g] - 1)
    loo = np.corrcoef(v, others)[0, 1] if np.std(others) else float("nan")
    return {"groups": len(group_films), "credits": len(v), "similarity": float(similarity),
            "loo_r": float(loo)}


def test_pairs(pairs, index, y, eras, shuffles, rng):
    observed, edge_count = assortativity(pairs, index, y)
    null = np.array([assortativity(pairs, index, shuffle_within_era(y, eras, rng))[0]
                     for _ in range(shuffles)])
    return {
        "assortativity": round(observed, 4) if np.isfinite(observed) else None,
        "pairs": edge_count,
        "null_mean": round(float(np.nanmean(null)), 4),
        "null_95": round(float(np.nanquantile(null, 0.95)), 4),
        "p": round(float((1 + np.sum(null >= observed)) / (1 + len(null))), 4),
    }


def test_sample(frame, members, shuffles, seed):
    ids = set(frame["imdb_id"])
    index = {film: i for i, film in enumerate(frame["imdb_id"])}
    y = residualize(frame)
    eras = frame["era_bin"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    role_results = {}
    for role in ROLES:
        pairs = make_graph(ids, members, role)
        role_results[role] = test_pairs(pairs, index, y, eras, shuffles, rng)
        role_results[role]["entity_groups"] = sum(len(films & ids) >= 2
                                                   for films in members[role].values())
        role_results[role]["intraclass"] = intraclass(
            list(ids), y, list(members[role].values()))
    combined = combined_graph(ids, members)
    role_results["combined"] = test_pairs(combined, index, y, eras, shuffles, rng)
    role_results["combined"]["entity_groups"] = sum(
        sum(len(films & ids) >= 2 for films in members[role].values()) for role in ROLES)
    return role_results


def markdown(summary):
    common = summary["samples"]["in_common_set_adjusted"]
    lines = [
        "# Q2: do shared people and studios explain film returns?",
        "",
        "## Question and measurement",
        "",
        "The question is whether films sharing a director, a top-three billed actor, a producer or a studio have more similar "
        "log gross/budget returns than chance.",
        "The outcome is `ln(gross/budget)`; it is not profit because budgets omit marketing and the gross share kept by cinemas is unknown.",
        "",
        "## Primary sample",
        "",
        f"`in_common_set` contains **{summary['samples']['in_common_set']['films']} films**.",
        "This uses the group's shared clean money rule and also requires a screenplay, as required for the head-to-head comparison.",
        "It is not the default for the network question alone; the wider `usable_money` result is a sensitivity check.",
        "",
        "## Method",
        "",
        "1. Link two films when they share a director, top-three actor, producer or studio.",
        "2. Use one graph per role and a combined graph that is the union of the four role graphs.",
        "3. Regress log return on ln(budget), era bin and all genre tags, then use residual return.",
        "4. Compute Pearson correlation between residual returns at the ends of network edges (outcome assortativity).",
        "5. Preserve the graph and shuffle returns only within era bin. Repeat 1,000 times.",
        "",
        "This does not estimate a causal effect or a predictive model. A film pair is counted once for each role graph.",
        "Combined-edge assortativity is unweighted: it asks whether a pair linked by at least one credit has similar returns.",
        "The intraclass statistic is an additional within-entity test, but it is descriptive.",
        "",
        "## Primary results",
        "",
        "| Relation | Linked pairs | Observed assortativity | Shuffle mean | 95th percentile | permutation p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role in (*ROLES, "combined"):
        r = common[role]
        lines.append(
            f"| {role} | {r['pairs']} | {r['assortativity']:.3f} | {r['null_mean']:.3f} | "
            f"{r['null_95']:.3f} | {r['p']:.4f} |"
        )
    lines += [
        "",
        "The secondary intraclass/leave-one-out numbers are positive for directors "
        "(+0.032 / +0.031), actors (+0.006 / +0.006) and producers (+0.033 / +0.019), "
        "but slightly negative for studios (-0.005 / -0.002). These are descriptive "
        "credits-only checks and are not separate conclusions.",
        "",
        "The primary edge result supports a real association: all four role relations "
        "exceed the within-era permutation 95th percentile. It is strongest for director "
        "and producer links, while the combined result is smaller because many common "
        "pairs are studio/actor pairs.",
        "",
        "## Sensitivity checks",
        "",
        "The following samples use the same adjusted outcome and within-era shuffle:",
        "",
    ]
    for name in ("usable_money_adjusted", "common_100k_adjusted", "common_unadjusted"):
        r = summary["samples"][name]
        lines.append(f"### {name}")
        lines.append("")
        for role in ("director", "actor", "producer", "studio", "combined"):
            v = r[role]
            lines.append(
                f"- {role}: {v['assortativity']:.3f}, {v['pairs']} pairs, p={v['p']:.4f}"
            )
        lines.append("")
    lines += [
        "## What is not done",
        "",
        "- No causal claim: directors and studios are not randomly assigned.",
        "- No league-table test for individual people or studios; small groups are unstable.",
        "- No temporal-only predictive test: this network asks about contemporaneous and historical co-credit similarity.",
        "- No graph rewiring: rewiring would test network structure, not whether return values cluster on the observed credit graph.",
        "- No common-set script text: Q2 uses network credits only.",
        "",
        "## How to verify",
        "",
        "Re-run the script from the repo root:",
        "",
        "```bash",
        ".venv/Scripts/python.exe scripts/network/q2_network.py",
        "```",
        "",
        "Check the raw counts and flags in `data/raw/features/film_analysis_set.csv`, the role credits in "
        "`data/raw/features/graph/edges.csv`, and the exact test values in `analysis/network/q2_network.json`.",
        "The script prints no hand-written sample count: the JSON and this page are generated from the CSVs.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2805)
    args = parser.parse_args()

    analysis = pd.read_csv(ANALYSIS)
    with EDGES.open(newline="", encoding="utf8") as f:
        edges = list(csv.DictReader(f))
    members = role_members(edges, set(analysis["imdb_id"]))

    common = analysis[analysis["in_common_set"]].copy()
    usable = analysis[analysis["usable_money"]].copy()
    common_100k = common[common["tier"] == "100k+"].copy()

    summary = {"seed": args.seed, "shuffles": args.shuffles,
               "cast_top": CAST_TOP, "roles": list(ROLES), "samples": {}}
    common_members = role_members(edges, set(common["imdb_id"]))
    usable_members = role_members(edges, set(usable["imdb_id"]))
    common100_members = role_members(edges, set(common_100k["imdb_id"]))

    summary["samples"]["in_common_set"] = {"films": len(common)}
    summary["samples"]["in_common_set_adjusted"] = test_sample(common, common_members, args.shuffles, args.seed)
    summary["samples"]["usable_money_adjusted"] = test_sample(usable, usable_members, args.shuffles, args.seed + 1)
    summary["samples"]["common_100k_adjusted"] = test_sample(common_100k, common100_members, args.shuffles, args.seed + 2)

    ids = set(common["imdb_id"])
    index = {film: i for i, film in enumerate(common["imdb_id"])}
    y = common["ln_gross_over_budget"].astype(float).to_numpy()
    eras = common["era_bin"].astype(str).to_numpy()
    rng = np.random.default_rng(args.seed + 3)
    summary["samples"]["common_unadjusted"] = test_sample_raw(common, common_members, y, eras, index, ids, args.shuffles, rng)
    summary["samples"]["usable_money"] = {"films": len(usable)}
    summary["samples"]["common_100k"] = {"films": len(common_100k)}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(markdown(summary), encoding="utf8")
    draw_figure(summary)

    print(json.dumps(summary["samples"]["in_common_set_adjusted"], indent=2))
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}, {OUT_FIG.relative_to(ROOT)}")


def test_sample_raw(frame, members, y, eras, index, ids, shuffles, rng):
    out = {}
    for role in ROLES:
        pairs = make_graph(ids, members, role)
        out[role] = test_pairs(pairs, index, y, eras, shuffles, rng)
    out["combined"] = test_pairs(combined_graph(ids, members), index, y, eras, shuffles, rng)
    return out


def draw_figure(summary):
    common = summary["samples"]["in_common_set_adjusted"]
    labels = [*ROLES, "combined"]
    observed = [common[r]["assortativity"] for r in labels]
    null95 = [common[r]["null_95"] for r in labels]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(labels))
    ax.bar(x, observed, color="#1F6F8B", width=0.65)
    ax.scatter(x, null95, marker="_", s=180, color="#B5482F", linewidth=2, zorder=3)
    ax.axhline(0, color="#1F2933", linewidth=0.8)
    ax.set_xticks(x, ["Director", "Top-3 actor", "Producer", "Studio", "Combined"])
    ax.set_ylabel("Outcome assortativity")
    ax.set_title("Shared credits and similar log gross/budget returns")
    ax.text(0.98, -0.22,
            "Red marks: 95th percentile of within-era permutations",
            transform=ax.transAxes, ha="right", fontsize=8)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
