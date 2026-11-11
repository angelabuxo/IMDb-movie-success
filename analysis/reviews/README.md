# Reviews analysis — do money, critics and audience agree?

Notebook: [`01_exploration.ipynb`](01_exploration.ipynb). Derived numbers/figures: [`outputs/`](outputs/).

## Sample (from data)

| Slice | N | Under budget |
|---|---:|---:|
| Clean 100k+, year ≤ 2017 | 2 095 | 130 (6.2%) |
| … with Tag Genome reviews | 2 070 | — |
| Optional 10k Wikidata ≤ 2017 | 1 012 | 316 (31.2%) |

Tag Genome has **no dates** and **no post-2017 coverage**, so audience-text work is cut at 2017.

`movie_level_table.csv` includes `disagreement_residual` (and `disagreement_group`, `profit_quintile`) for the network branch to join on `imdb_id` (non-null for the 2 095 primary movies).

## Idea 1 — scores and residual disagreement

**Null.** Primary: shuffle within **genre × decade** (first-listed genre). Report ρ, stratified null mean/std, **z-score**. Global shuffle is secondary (`p_null_global`).

| Pair | N | ρ | null_mean_strat | null_std_strat | z_strat | p_null_strat | p_null_global |
|---|---:|---:|---:|---:|---:|---:|---:|
| log(g/b) ↔ IMDb | 2 095 | 0.266 | 0.099 | 0.018 | 9.51 | 0.001 | 0.001 |
| log(g/b) ↔ Metacritic | 1 834 | 0.261 | 0.089 | 0.020 | 8.71 | 0.001 | 0.001 |
| log(g/b) ↔ RT | 2 093 | 0.326 | 0.122 | 0.018 | 11.41 | 0.001 | 0.001 |
| IMDb ↔ Metacritic | 1 834 | 0.757 | 0.233 | 0.018 | 29.14 | 0.001 | 0.001 |
| IMDb ↔ RT | 2 093 | 0.785 | 0.234 | 0.018 | 31.29 | 0.001 | 0.001 |
| Metacritic ↔ RT | 1 833 | 0.931 | 0.218 | 0.019 | 38.00 | 0.001 | 0.001 |

Stratified null means are **not** ~0: genre×decade structure alone induces positive rank correlation. Observed ρ still sits many σ above that baseline.

**Disagreement.** Residual of `rank(IMDb) ~ rank(log(gross/budget))`.

| Quantity | Value |
|---|---:|
| N with residual | 2 095 |
| Residual mean (numerical ~0) | ≈ 0 |
| Residual std | 582.7 |
| rating_above_money (residual Q5) | 419 |
| rating_below_money (residual Q1) | 419 |
| middle | 1 257 |
| Profit Q1 N / max log(g/b) | 420 / 0.601 |
| Profit Q5 N / min log(g/b) | 419 / 1.983 |
| Mean IMDb in profit Q1 / Q5 | 6.78 / 7.29 |

Top positive residual (liked beyond money): *It's a Wonderful Life*, *Once Upon a Time in America*, *Once Upon a Time in the West*, …  
Top negative residual (liked less than money): *Fifty Shades of Grey*, *Twilight: New Moon*, *Annabelle*, …

Display uses **extreme profit quintiles**, not break-even × median. Figures: `fig_scatter_money_scores.png`, `fig_profit_quintiles_disagreement.png`.

## Text features

| Check | N | ρ | null_mean_strat | z_strat | p_null_strat | p_null_global |
|---|---:|---:|---:|---:|---:|---:|
| VADER ↔ log(g/b) | 2 070 | 0.127 | 0.025 | 6.10 | 0.001 | 0.001 |
| VADER ↔ IMDb | 2 070 | 0.218 | 0.054 | 10.28 | 0.001 | 0.001 |
| Diversity ↔ log(g/b) | 2 070 | −0.160 | −0.053 | −5.86 | 0.001 | 0.001 |
| Length ↔ log(g/b) | 2 070 | −0.105 | −0.065 | −2.51 | 0.006 | 0.001 |

Hand-label sample (50 reviews) still awaiting labels. VADER vs Stanford polarity: accuracy **0.733** on 1 988 non-neutral of 2 000 docs.

## Idea 2 — Tag Genome language of disagreement groups

Compares Tag Genome reviews for `rating_above_money` vs `rating_below_money` (not Wikipedia vs audience).

| Coverage | Value |
|---|---:|
| Movies with text (above / below) | 415 / 412 |
| Min term count (both corpora) | 25 |
| Terms kept | 4 825 |
| Name tokens banned | 4 381 |

Divergent examples (from data): *tarantino*, *noir*, *nomination* favor rating_above_money; *scares*, *slasher*, *doll* favor rating_below_money.

Figure: `fig_scattertext_disagreement_groups.png`.

## Re-run

```bash
PYTHONPATH=. .venv/bin/python -m analysis.reviews.run_exploration
```

## Open

1. Hand-label the 50-review sample and report agreement.
2. Hand-check 10k-tier money if used beyond sensitivity.
3. Franchise / awards leakage (project-level).
