# Q2: do shared people and studios explain film returns?

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

The secondary intraclass/leave-one-out numbers are positive for directors (+0.032 / +0.031), actors (+0.006 / +0.006) and producers (+0.033 / +0.019), but slightly negative for studios (-0.005 / -0.002). These are descriptive credits-only checks and are not separate conclusions.

The primary edge result supports a real association: all four role relations exceed the within-era permutation 95th percentile. It is strongest for director and producer links, while the combined result is smaller because many common pairs are studio/actor pairs.

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

## Community view

The film-only explorer now compares Louvain, Leiden, Infomap and label propagation on the
weighted film-projection graph inside the current filters. Louvain optimizes modularity,
Leiden improves that search with a refinement step, Infomap minimizes map/flow description
length rather than maximizing modularity, and label propagation is fast and can be unstable.
The explorer reports each method's community count and modularity, but modularity is a
diagnostic for Infomap rather than its objective.

Each community table reports film count, sum, mean, median, minimum, maximum and range of
`ln(gross/budget)`. It also marks missing return values rather than inventing zeros. The
community partitions use the untruncated weighted graph among displayed films; the edge
slider only limits which links are drawn.

## Post-hoc robustness runs on full IMDb careers

Declared on 23 November 2026, before either run. Both are post-hoc: they were chosen after seeing the
main results, to check whether counting careers only through films with 10,000+ IMDb votes changes them.
Every result below is reported, whichever way it falls.

**1. Experience counted from full careers** (`analysis/compare/experience_full_imdb.py`).
`film_network_features.csv` counts a person's earlier films only among films with 10,000+ votes. The
rebuild counts every earlier non-adult film (`titleType` movie) the person is credited on in IMDb's full
`title.principals` and `title.basics`. Teams are the same people as before: directors, writers, producers
and the first three billed actors. Studios and distributors aren't in IMDb, so their experience stays as it
was. Track records (mean ln(gross/budget) of earlier films) need money, which only the 10,000-vote films
have, so they stay as they were. The test is the one used for the genre specialists
(`q1_before_release.py`): on the 6,346 `usable_money` films, the OLS coefficient per SD of each measure on
ln(gross/budget) net of budget decile x era, with genre dummies and the sequel flag, and a p-value from
1,000 shuffles of the measure within first genre x budget decile, the model refitted every time. Both
versions of each experience count are tested, plus the unchanged track records for comparison. The
out-of-time ridge R² (−0.009 → 0.192) is dropped from the paper rather than recomputed, because ridge is
not a course method.

**2. Communities of the people network built from full IMDb**
(`analysis/compare/network_communities.py --links full_imdb`). Films: every non-adult film in
`title.principals` with at least one link (656,445 in the run; the 723,734 first written here, from a sizing
run, counted adult films and films with no link). Links: two films share a director, one of the first three
billed actors or actresses, or a producer, weighted by the number of shared people. No studio links,
because IMDb doesn't record studios. Methods as in the 6,346-film run: Louvain over 10 seeds (best
modularity kept, mean NMI to the other seeds), a null of 50 degree-preserving rewirings with shuffled
weights and the same 10 seeds each, and NMI with each film's rarest IMDb genre against 1,000 genre shuffles.
Louvain runs in igraph (`community_multilevel`), the same algorithm as networkx's, because networkx takes
about 85 minutes per run on 23 million links. Returns are tested only on the 6,346 `usable_money` films:
every community with at least 30 of them, against the budget-decile shuffle (10,000 shuffles, Holm), and
every such community is reported. To separate dropping studios from adding unnoticed films, the
6,346-film network is also rerun without studio links (`--links no_studio`). Together these answer whether
people alone, without the studio, form the "production houses".

**3. Communities inside the full network's Hollywood community**
(`analysis/compare/network_communities_hollywood.py`). Declared on 23 November 2026 after run 2 showed one
community holding 136,577 films and 5,821 of the 6,346 money films, and before running. Modularity has a resolution
limit: in a network of 656,445 films, a group the size of Blumhouse's (108 money films in the 6,346-film run) can be
too small for Louvain to split off, so its absence in run 2 doesn't show it isn't there. The run takes that largest
community's films and links as they are in run 2 and repeats run 2's methods on them alone: Louvain over 10 seeds,
50 degree-preserving rewirings with the same seeds, NMI with the rarest genre against 1,000 shuffles, and the
budget-decile shuffle (10,000 shuffles, Holm) for every sub-community with at least 30 money films, all reported.
It also reports where the films of each 10,000-vote community fall (the share in its most common sub-community),
Blumhouse's first.

Leiden and Infomap are removed from the community runs. They aren't course methods, and Louvain's
agreement across seeds already measures stability.

## How to verify

Re-run the script from the repo root:

```bash
.venv/Scripts/python.exe scripts/network/q2_network.py
```

Check the raw counts and flags in `data/raw/features/film_analysis_set.csv`, the role credits in `data/raw/features/graph/edges.csv`, and the exact test values in `analysis/network/q2_network.json`.
The script prints no hand-written sample count: the JSON and this page are generated from the CSVs.
