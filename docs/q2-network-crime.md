# Q2: do shared people and studios explain film returns?

**Genre scope:** `Crime`.

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
| director | 248 | 0.064 | -0.009 | 0.101 | 0.1189 |
| actor | 746 | 0.094 | -0.005 | 0.054 | 0.0060 |
| producer | 427 | 0.207 | -0.005 | 0.078 | 0.0010 |
| studio | 953 | 0.058 | -0.006 | 0.051 | 0.0340 |
| combined | 1948 | 0.067 | -0.005 | 0.031 | 0.0020 |

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

- director: 0.133, 1246 pairs, p=0.0010
- actor: 0.045, 6576 pairs, p=0.0020
- producer: 0.144, 3563 pairs, p=0.0010
- studio: 0.090, 9319 pairs, p=0.0010
- combined: 0.071, 17776 pairs, p=0.0010

### common_100k_adjusted

- director: 0.011, 161 pairs, p=0.3516
- actor: 0.058, 411 pairs, p=0.0889
- producer: 0.141, 250 pairs, p=0.0240
- studio: 0.059, 556 pairs, p=0.0759
- combined: 0.049, 1084 pairs, p=0.0430

### common_unadjusted

- director: 0.074, 248 pairs, p=0.1359
- actor: 0.080, 746 pairs, p=0.0180
- producer: 0.213, 427 pairs, p=0.0010
- studio: 0.048, 953 pairs, p=0.0470
- combined: 0.059, 1948 pairs, p=0.0110

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
