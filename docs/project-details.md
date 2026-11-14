# Project details

The one-page version is [overview.md](overview.md). This page gives each analysis its own short block:
what it asks, where it lives, and what it found. State on 14 November 2026.

## Data and shared rules

| What | Count | Source |
|---|---:|---|
| Movies with 10,000+ votes (the graph) | 12,673 | IMDb |
| Clean films with budget and gross in USD | 2,446 (100k+ votes), 6,851 (10k+) | Wikipedia infoboxes |
| Screenplays | 1,095 clean 100k+ films | MovieSum |
| Audience reviews | 2,070 clean 100k+ films, none after 2017 | MovieLens Tag Genome |
| Critic scores | Metacritic 2,431, Rotten Tomatoes 2,740 | Wikipedia, Wikidata |

"Clean" means budget and gross both in USD and at least $1,000, no rentals-only or regional gross, no
streaming distributor, and released before 2026. The 100k+ tier favours hits: only 169 of its 2,446 films
grossed less than their budget. Sources and licences are in [data-sources.md](data-sources.md).

## Who made the film

**Movie graph** (`scripts/network/`, `notebooks/movie_graph.ipynb`). Movies, people (by role and billing)
and companies in one network. Studio names are merged through Wikidata IDs.
- The studio's track record is the strongest single measure: ρ +0.27 with the multiple, net of budget and
  year, on 4,568 films.
- All network measures together lift R² from 0.041 to 0.130 over budget, year and genre (shuffle
  p = 0.005).
- A director's next film is a hit 35% of the time after a hit, and 18% after a non-hit.

**Shared-people returns** (`scripts/network/roi_similarity.py`). Do films with the same director,
producer, cast member or studio have more similar returns than chance?
- Every role clusters: 0.09 (top-three cast) to 0.17 (producers), against a shuffle null of at most 0.02.
- It survives dropping franchise films, ranking the returns, and keeping only films from 1980 on.
- About half survives a shuffle within each studio.
- It's descriptive: it uses all years at once.

**Actor network** (Niklas, branch `initial_actor_network`, not merged). Top-three billed actors, linked by
shared films. It comes with an interactive page with projections, communities, centrality and per-actor ROI.

## What the film says

**Scripts and reviews notebook** (`notebooks/scripts_and_reviews.ipynb`). The first pass over the text.
- Only 2 of 15 script features pass q < 0.05 net of budget and year. The strongest is a character network
  centred on one character (ρ −0.097).
- Six VADER arc shapes don't differ in profit (p = 0.99).

**Scattertext words** (`analysis/scripts_text/`, [scripts-text.md](scripts-text.md)). Pre-registered. Top
against bottom quarter by multiple, within genre and decade, on dialogue.
- The difference is slightly above random groupings (z = 2.08, p = 0.024).
- It disappears without franchise films (z = 0.50). Dropping the same number of random films never does
  that (0 of 50 tries), so the signal sits in the franchise films themselves.
- A word classifier doesn't beat genre, decade and budget (AUC 0.58 against 0.58).

**Emotional arcs** (`analysis/arcs/`, [scripts-arcs.md](scripts-arcs.md)). Pre-registered, with one test.
Every sentence of 834 non-franchise scripts is scored by DistilBERT SST-2. Each film becomes a 100-point
tone curve, summarised as swing (D1) and ending lift (D2). Budget, genre, decade and scene length are
controlled.
- Swing: no link (ρ +0.035, p 0.32).
- Ending lift: a near miss (ρ +0.078, p 0.028, against a pre-set cutoff of 0.025). Its sign holds in 12 of
  13 variants, but not with VADER.
- The arc adds nothing to out-of-sample prediction.
- About half of the arc variation between films is scene noise (split-half reliability around 0.45).
- A fourth sentiment model, ModernBERT, is running as a robustness check.

## How it was received

**Reviews exploration** (Àngela, branch `reviews-exploration`, not merged). 2,095 films up to 2017. Every
correlation is tested against a shuffle within genre × decade.
- The multiple goes with IMDb rating at ρ 0.27, Metacritic 0.26 and Rotten Tomatoes 0.33. Genre and decade
  alone give about 0.1.
- The critics agree with each other far more (Metacritic and Rotten Tomatoes at 0.93).
- Review tone goes with the multiple at 0.13.
- A disagreement score (rating rank minus money rank) finds "liked beyond their money" films (*tarantino*,
  *noir*) and "earned beyond their liking" films (*scares*, *slasher*).

## What predicts profit

`notebooks/what_predicts_profit.ipynb`. Trained on films to 2010, tested on 2011 to 2017.

| Model | Out-of-sample R² |
|---|---:|
| Budget, year, genre | −0.010 |
| + network | 0.202 |
| All pre-release blocks (adds month, franchise) | 0.244 |
| + critic and audience scores | 0.312 |

- On films with a screenplay, script features add nothing beyond the network (R² 0.037 against 0.036).
- Director hit streaks beat a shuffle of each director's own films (+0.175 against +0.090).
- PageRank adds nothing beyond plain experience.
- Rotten Tomatoes goes with the multiple at ρ +0.25. There are 158 loved flops and 205 panned hits.

## Open and weak spots

- **Human validation**, which only people can fill:
  - `analysis/scripts_text/outputs/validation_*.csv`;
  - `analysis/arcs/validation/` and `analysis/arcs/outputs/s12_back_to_text.csv`;
  - the 50-review sample on `reviews-exploration`.
- **Arcs Step 11** waits on the instrument question (whole-review VADER against sentence-level VADER).
- **Weak spots**: the franchise flag is rough, release month isn't checked, reviews stop at 2017, and gross
  over budget isn't profit.

## Where things are

| Path | What |
|---|---|
| `scripts/` | Data download and parsing, the graph, network features |
| `analysis/scripts_text/`, `analysis/arcs/` | The two pre-registered text analyses: config, LOG, steps, outputs |
| `notebooks/` | Graph, profit model, scripts and reviews, plus one folder per text analysis |
| `site/scripts/`, `site/arcs/` | Static pages for the text analyses |
| `data/bundles/` | Compressed data that ships with the repo |
