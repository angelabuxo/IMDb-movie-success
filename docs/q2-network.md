# Q2: do shared people and studios explain film returns?

**Genre scope:** `any`.

## Question and measurement

The question is whether films sharing a director, a top-three billed actor, a producer or a studio have more similar log gross/budget returns than chance.
The outcome is `ln(gross/budget)`; it is not profit because budgets omit marketing and the gross share kept by cinemas is unknown.

## Primary sample

`in_common_set` contains **1598 films**.
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
| director | 2311 | 0.090 | -0.001 | 0.035 | 0.0010 |
| actor | 9958 | 0.064 | -0.002 | 0.014 | 0.0010 |
| producer | 6639 | 0.091 | -0.001 | 0.020 | 0.0010 |
| studio | 20990 | 0.048 | -0.001 | 0.010 | 0.0010 |
| combined | 33685 | 0.048 | -0.001 | 0.008 | 0.0010 |

The primary edge result supports a real association: all four role relations exceed the within-era permutation 95th percentile. The secondary intraclass/leave-one-out numbers are descriptive and range from -0.043 to -0.009 for the common set.

## Genre-specific combined results

| Genre | Films | Linked pairs | Assortativity | 95th shuffle percentile | Permutation p |
|---|---:|---:|---:|---:|---:|
| Drama | 882 | 11,212 | +0.035 | +0.014 | 0.0010 |
| Comedy | 541 | 4,272 | +0.016 | +0.020 | 0.0919 |
| Action | 429 | 4,418 | +0.048 | +0.019 | 0.0010 |
| Adventure | 338 | 2,374 | +0.046 | +0.028 | 0.0100 |
| Crime | 338 | 1,948 | +0.067 | +0.031 | 0.0020 |
| Thriller | 297 | 1,213 | +0.056 | +0.041 | 0.0150 |
| Horror | 200 | 534 | +0.182 | +0.059 | 0.0010 |
| Supernatural | 0 | No linked graph | No linked graph | No linked graph | No linked graph |
| Horror or Supernatural | 200 | 534 | +0.182 | +0.059 | 0.0010 |

Each genre also has exact role-level, sensitivity and intraclass results:
[Action](q2-network-action.md), [Adventure](q2-network-adventure.md), [Comedy](q2-network-comedy.md),
[Crime](q2-network-crime.md), [Drama](q2-network-drama.md), [Thriller](q2-network-thriller.md).
[Horror](q2-network-horror.md), [Supernatural](q2-network-supernatural.md),
and [Horror or Supernatural](q2-network-horror-or-supernatural.md).

## Sensitivity checks

The following samples use the same adjusted outcome and within-era shuffle:

### usable_money_adjusted

- director: 0.116, 16156 pairs, p=0.0010
- actor: 0.042, 92131 pairs, p=0.0010
- producer: 0.116, 62143 pairs, p=0.0010
- studio: 0.061, 231646 pairs, p=0.0010
- combined: 0.055, 353777 pairs, p=0.0010

### common_100k_adjusted

- director: 0.120, 1503 pairs, p=0.0010
- actor: 0.047, 5664 pairs, p=0.0010
- producer: 0.135, 3865 pairs, p=0.0010
- studio: 0.036, 13158 pairs, p=0.0010
- combined: 0.038, 20039 pairs, p=0.0010

### common_unadjusted

- director: 0.107, 2311 pairs, p=0.0010
- actor: 0.056, 9958 pairs, p=0.0010
- producer: 0.093, 6639 pairs, p=0.0010
- studio: 0.063, 20990 pairs, p=0.0010
- combined: 0.052, 33685 pairs, p=0.0010

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
