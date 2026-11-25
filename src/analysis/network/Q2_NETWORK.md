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

Result (job run 23 November 2026, 10 seeds, 50 rewirings, 10,000 shuffles). The Hollywood community's 136,577 films
and 2,394,367 links split into 55 sub-communities with modularity 0.595, against a rewired mean of 0.126 and maximum
of 0.126 (p = 0.0196, the smallest 50 rewirings allow). The split is less stable than run 2's: the best seed agrees
with the other nine at NMI 0.505. NMI with the rarest genre is 0.016 against a null 95th percentile of 0.001.
25 sub-communities hold at least 30 money films (5,615 of the 5,821 inside), and 7 differ from films of the same
budget after Holm:

| Sub-community (top studios) | Money films | Excess | Holm p |
|---|---|---|---|
| Blumhouse; Di Bonaventura; Vertigo | 299 | +0.431 | 0.0025 |
| Imagine; Amblin; Malpaso | 232 | +0.351 | 0.0025 |
| Kingdom Story; TriStar; Sherwood (faith films) | 40 | +0.884 | 0.026 |
| Dog Eat Dog; Broken Lizard (Michael Moore, Jackass) | 31 | +0.971 | 0.027 |
| Emmett/Furla/Oasis; Tribeca | 308 | -0.318 | 0.0025 |
| Thunder Road; Constantin | 141 | -0.396 | 0.013 |
| Columbia; Point Grey; Hydraulx | 55 | -0.842 | 0.0025 |

Of the 10,000-vote network's 16 communities, Blumhouse's stays together: 76% of its 105 films here fall in one
sub-community. Marvel's 60 films do too (68%), but inside the largest sub-community. The others spread out, with
23% to 52% in their most common sub-community. NMI with the 10,000-vote communities on 5,796 shared money films is
0.143, against 0.092 for run 2 as a whole. All rows are in
`analysis/compare/outputs/communities_hollywood/` (`community_profiles.csv`, `houses_10k.csv`).

What it shows: the houses that earn most, Blumhouse and the Imagine/Amblin group, come back from people alone once the
national industries are set aside, with smaller gaps than with studio links (Blumhouse +1.06 there). Unlike run 1,
groups that fall short also pass, so losing houses are visible here too. The low seed agreement means individual
group boundaries, especially for the small groups, shouldn't be read closely.

**4. A track record from IMDb alone** (`analysis/compare/notice_track_imdb.py`). Declared on 23 November 2026,
before running. Money track records see only films with 10,000+ votes, so a person's unnoticed flops never enter them,
and no new money source is used. IMDb has votes and a rating for every film, so for the same teams as run 1 this
builds, over every earlier non-adult IMDb film they worked on, the share that stayed under 1,000 votes (no rating
counts as under) and the mean rating of the rated ones, averaged over each role's people. Both are tested like the
money track records (OLS per SD, 1,000 shuffles within first genre x budget decile, 6,346 `usable_money` films), and
each is correlated (Spearman) with the role's money track record as a second method. Votes and ratings are counted
today, not at release, and they are not money. Every result is reported.

Result (job run 23 November 2026, 1,000 shuffles). Coefficients are per SD of the measure; the money track records
reproduce the committed ones exactly.

| Role | Share of earlier films under 1,000 votes | Mean rating of earlier films | Spearman with money track (share / rating) |
|---|---|---|---|
| director | +0.025 (p 0.18) | +0.062 (p 0.004) | 0.016 / 0.047 |
| writer | -0.019 (p 0.36) | +0.036 (p 0.065) | -0.019 / 0.093 |
| producer | -0.085 (p 0.001) | +0.129 (p 0.001) | -0.123 / 0.185 |
| lead actors | +0.022 (p 0.32) | +0.025 (p 0.16) | 0.047 / 0.018 |

What it shows: producers leave the clearest trace. A producer whose earlier films went unnoticed, or were rated low,
does worse with the budget, and well-rated directors do better. The effects are about a third to half the size of the
money track records (+0.28 for producers), and they barely correlate with them, so the money record isn't just a
noticed past. Votes and ratings are counted today, so a film revived later counts as noticed. All rows are in
`analysis/compare/outputs/notice_track_imdb/`.

**5. How far missing flops could inflate track records** (`analysis/compare/track_record_bounds.py`).
Declared on 23 November 2026, after run 1 and before running this one. Track records average the returns of
a person's earlier films, but we have money only for the 12,673 graph films, so their unnoticed earlier films,
mostly flops, never enter. Two checks, with the same OLS and shuffle as run 1, on the directors, writers,
producers and first three billed actors (studios and distributors aren't in IMDb):
(a) a pessimistic bound: every earlier IMDb film of the team with no money figure is given a bad return, once
the mean return of the films Wikipedia lacks that TMDB has money for (−0.379, `analysis/genre/outputs/tmdb`)
and once the mean of the bottom quarter of all clean returns. If track records still pass, missing flops
can't be what drives them. (b) a coverage split: track records tested separately on films whose team's earlier
IMDb films mostly have money and on films where they mostly don't. If missing flops inflate track records,
the effect should be larger where coverage is low. Every result is reported.

(Declared as run 3 on the branch `g/10k-vote-cut-followups-ab454f`, commit 97b957d, and brought here on 25 November 2026. It is numbered 5 here because runs 3 and 4 were taken.)

Result (job run 23 November 2026, 1,000 shuffles, commit 366cbc7). Coefficients are per SD; every shuffle p is 0.001.

| Role | Rebuilt | Missing films at -0.379 | Missing films at -1.249 | Coverage at or above median | Coverage below median |
|---|---|---|---|---|---|
| director | +0.192 | +0.169 | +0.155 | +0.180 | +0.204 |
| writer | +0.212 | +0.201 | +0.179 | +0.202 | +0.229 |
| producer | +0.273 | +0.266 | +0.252 | +0.131 | +0.349 |
| lead actors | +0.154 | +0.140 | +0.117 | +0.169 | +0.159 |

What it shows: track records survive a pessimistic guess about the films without money. Producers' records are much stronger where fewer of their earlier films have money, so part of theirs may be what is missing. It is a stress test: the true returns of those films are unknown, and one value for all of them also mixes in how many films a person made.

Leiden and Infomap are removed from the community runs. They aren't course methods, and Louvain's
agreement across seeds already measures stability.

## How to verify

Re-run the script from the repo root:

```bash
.venv/Scripts/python.exe scripts/network/q2_network.py
```

Check the raw counts and flags in `data/raw/features/film_analysis_set.csv`, the role credits in `data/raw/features/graph/edges.csv`, and the exact test values in `analysis/network/q2_network.json`.
The script prints no hand-written sample count: the JSON and this page are generated from the CSVs.
