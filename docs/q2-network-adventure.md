# Q2: do shared people and studios explain film returns?

**Genre scope:** `Adventure`.

## Question and measurement

The question is whether films sharing a director, a top-three billed actor, a producer or a studio have more similar log gross/budget returns than chance.
The outcome is `ln(gross/budget)`; it is not profit because budgets omit marketing and the gross share kept by cinemas is unknown.

## Primary sample

`in_common_set` contains **338 films**.
This uses the group's shared clean money rule and also requires a screenplay, as required for the head-to-head comparison.
It is not the default for the network question alone; the wider `usable_money` result is a sensitivity check.

## Method

1. Link two films when they share a director, top-three actor, producer or studio.
2. Use one graph per role and a combined graph that is the union of the four role graphs.
3. Regress log return on ln(budget), era bin and all genre tags, then use residual return.
4. Compute Pearson correlation between residual returns at the ends of network edges (outcome assortativity).
5. Preserve the graph and shuffle returns only within era bin. Repeat 1,000 times.

This does not estimate a causal effect or a predictive model. A film pair is counted once for each role graph.
Combined-edge assortativity is unweighted: it asks whether a pair linked by at least one credit has similar returns.
The intraclass statistic is an additional within-entity test, but it is descriptive.

## Primary results

| Relation | Linked pairs | Observed assortativity | Shuffle mean | 95th percentile | permutation p |
|---|---:|---:|---:|---:|---:|
| director | 223 | 0.213 | -0.005 | 0.110 | 0.0050 |
| actor | 500 | 0.063 | -0.008 | 0.068 | 0.0649 |
| producer | 454 | 0.075 | -0.008 | 0.073 | 0.0450 |
| studio | 1831 | 0.045 | -0.004 | 0.036 | 0.0260 |
| combined | 2374 | 0.046 | -0.005 | 0.028 | 0.0100 |

## Genre views

- [All genres](../analysis/network/q2_network.json)
- [Action](q2-network-action.md)
- [Adventure](q2-network-adventure.md)
- [Comedy](q2-network-comedy.md)
- [Crime](q2-network-crime.md)
- [Drama](q2-network-drama.md)
- [Thriller](q2-network-thriller.md)

Genre files are generated with `--genre <Genre>` or all at once with `--all-genres`.

## Sensitivity checks

The following samples use the same adjusted outcome and within-era shuffle:

### usable_money_adjusted

- director: 0.223, 1445 pairs, p=0.0010
- actor: 0.034, 4633 pairs, p=0.0100
- producer: 0.143, 4269 pairs, p=0.0010
- studio: 0.044, 21606 pairs, p=0.0010
- combined: 0.043, 27405 pairs, p=0.0010

### common_100k_adjusted

- director: 0.168, 196 pairs, p=0.0120
- actor: 0.116, 392 pairs, p=0.0140
- producer: 0.165, 379 pairs, p=0.0030
- studio: 0.055, 1460 pairs, p=0.0180
- combined: 0.064, 1879 pairs, p=0.0020

### common_unadjusted

- director: 0.226, 223 pairs, p=0.0020
- actor: 0.122, 500 pairs, p=0.0120
- producer: 0.137, 454 pairs, p=0.0030
- studio: 0.094, 1831 pairs, p=0.0040
- combined: 0.078, 2374 pairs, p=0.0010

## What is not done

- No causal claim: directors and studios are not randomly assigned.
- No league-table test for individual people or studios; small groups are unstable.
- No temporal-only predictive test: this network asks about contemporaneous and historical co-credit similarity.
- No graph rewiring: rewiring would test network structure, not whether return values cluster on the observed credit graph.
- No common-set script text: Q2 uses network credits only.

## How to verify

Re-run the script from the repo root:

```bash
.venv/Scripts/python.exe scripts/network/q2_network.py
```

Check the raw counts and flags in `data/raw/features/film_analysis_set.csv`, the role credits in `data/raw/features/graph/edges.csv`, and the exact test values in `analysis/network/q2_network.json`.
The script prints no hand-written sample count: the JSON and this page are generated from the CSVs.
