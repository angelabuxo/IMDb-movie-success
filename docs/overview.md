# Project overview: everything we're doing, in one place

Log-Log Legends (Àngela Buxó, Gyula Kürthy, Niklas Johansen), DTU 02805, autumn 2026.
State as of 14 November 2026. Paper due Friday 27 November, presentation Wednesday 2 December.

## The question

What makes a movie highly profitable, and does making money agree with being liked?

The course wants one dataset with both networks and text, checked measurements, and one story. So the
work splits into three strands: who made the film (network), what the film says (screenplays), and how it
was received (critics and audiences). Each strand asks what it adds to profit, and the reception strand
also asks whether money and liking agree.

## How the pieces fit

```
                       ┌──────────────────── DATA (data/raw/, never in git) ────────────────────┐
                       │ IMDb files · Wikipedia infoboxes and text · Wikidata · MovieSum scripts │
                       │ MovieLens Tag Genome reviews · Oscar data                               │
                       └───────────────┬────────────────────────────────────────────────────────┘
                                       │  scripts/get_data.py, data/bundles/*.tar.zst
                                       ▼
               shared tables: movies_features.csv, the clean sample, ln(gross / budget)
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
   NETWORK strand                 SCREENPLAY strand               RECEPTION strand
   who made the film              what the film says              how it was received
   ─────────────                  ─────────────────               ───────────────────
   movie graph                    scripts and reviews notebook    reviews exploration (Àngela, branch)
   actor network (Niklas, branch) Scattertext word analysis       critic vs audience vs money
   ROI similarity                 emotional arcs
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       ▼
                     WHAT PREDICTS PROFIT: every block side by side,
                     trained on films up to 2010, tested on 2011 to 2017
                                       ▼
                         paper (PNAS, 6 pages) · 3 slides · reflection
```

## Shared ground rules

These hold in every strand, so results can be put side by side.

- **Profit** is ln(worldwide gross / budget), "the multiple". It isn't real profit: marketing isn't in the
  budget and cinemas keep part of the gross. Niklas's actor network uses (gross − budget) / budget, the same
  idea on a different scale.
- **Clean sample**: budget and gross both in USD and at least $1,000, no rentals-only or regional grosses,
  no streaming distributors, released before 2026. That leaves 2,446 films with 100,000+ IMDb votes, or
  6,851 with 10,000+. The 100k+ tier favours hits (only 169 of 2,446 grossed less than their budget), while the
  10k+ tier has far more flops.
- **No leakage.** Network measures for a film use only films released before it. The main profit model is
  tested on later films than it was trained on.
- **Every result gets a check**: a shuffle null, a second method, or a hand-labelled sample.
- **Controls**: budget, year or decade, and genre are held fixed almost everywhere. Franchise films get a
  separate check, because they dominate the top of the profit list.
- **Copyright and terms.** No screenplay text, review text, raw data or IMDb fields go into git. Committed
  tables identify films by MovieSum name with genre and decade as broad groups. Sources whose terms forbid
  scraping (Box Office Mojo, The Numbers) are not used.

## The data

Where it comes from: [data-sources.md](data-sources.md). How it fits together: [imdb-data-structure.md](imdb-data-structure.md).

| What | Count | Source |
|---|---:|---|
| Movies with 100,000+ votes | 2,764 | IMDb |
| Movies with 10,000+ votes (graph) | 12,673 | IMDb |
| Budget and gross in USD | 2,567 of the 100k+ | Wikipedia infoboxes |
| Critic scores | Metacritic 2,431, Rotten Tomatoes 2,740 | Wikipedia and Wikidata |
| Screenplays | 1,181 (1,095 in the clean 100k+ sample) | MovieSum |
| Audience reviews | 2,070 clean 100k+ films, none after 2017 | MovieLens Tag Genome |
| Franchise flag | rough | Wikidata (follows, followed by, part of a series) |

Most of it comes with the repo as compressed bundles in `data/bundles/`. IMDb's raw files and the reviews
are downloaded with `scripts/get_data.py`.

## Strand 1: who made the film (network)

### The movie graph (on main)

One network of the 12,673 movies, the people who made them (actors with billing order, directors, writers,
producers, composers, cinematographers, editors) and the companies that produced and distributed them.
Studio names are merged through Wikidata IDs, so "Miramax" and "Miramax Films" are one node.

- Code: `scripts/network/`. Notebook: `notebooks/movie_graph.ipynb`.
- Findings:
  - The studio's track record is the strongest single network measure: ρ +0.27 with the multiple, net of
    budget and year, on 4,568 films. All 11 network measures pass q < 0.05.
  - Together the network measures raise R² from 0.041 to 0.130 over budget, year and genre (shuffle
    p = 0.005).
  - A director's next film is a hit 35% of the time after a hit, 18% after a non-hit.

### Do films that share people have similar returns? (on main)

For each role, do films with the same director, producer, cast member, studio and so on have more similar
returns than chance?

- Code: `scripts/network/roi_similarity.py`. Results: `analysis/network/roi_similarity.json`.
- Findings:
  - Every role clusters. After budget, decade and genre, similarity runs from 0.09 (top-three cast) to
    0.17 (producers), against a null of at most 0.02.
  - It holds without franchise films, with ranked returns, from 1980 on, and in the 100k+ tier.
  - Shuffling only within each studio, roughly half of the people roles' similarity survives.
- Limits: it is descriptive (it uses all years at once), and some of the clustering is how old grosses
  were reported.

### Actor co-appearance network (Niklas, branch `initial_actor_network`, not merged)

Actors are linked when they appear among a film's top three billed, weighted by shared films (10,000+
votes). It comes with an interactive HTML page: hover highlighting, director and movie projections,
community detection, centrality and per-actor ROI.

- Code: `scripts/imdb/build_actor_network*.py`, `scripts/network/actor_roi.py`.
- Status: ready to merge. Its README edits overlap with main's, so expect a small conflict.

## Strand 2: what the film says (screenplays)

### Scripts and reviews notebook (on main)

The group's first pass over the text: hit against flop scripts on character networks and text measures,
VADER sentiment arcs clustered into six shapes, distinctive words, and script embeddings against profit.

- Notebook: `notebooks/scripts_and_reviews.ipynb`.
- Findings:
  - Only 2 of 15 script features pass q < 0.05 net of budget and year. The strongest is how centred the
    character network is on one character (ρ −0.097).
  - Arc shapes don't differ in profit (Kruskal-Wallis p = 0.987).

### Scattertext: do profitable and unprofitable scripts use different words? (on main)

Within genre and decade, the top quarter of films by multiple against the bottom quarter, on dialogue. The
primary analysis was fixed before any results were seen.

- Code: `analysis/scripts_text/`. Notebooks: `notebooks/scripts_text/01` to `12`. Page:
  `site/scripts/index.html`. How to run it: [scripts-text.md](scripts-text.md).
- Findings, a negative result with its checks:
  - The dialogue of the two groups differs only slightly more than random groupings would (z = 2.08,
    p = 0.024).
  - The difference disappears without franchise films (z = 0.50, p = 0.30). Dropping the same number of
    random films never does that (0 of 50 tries).
  - A TF-IDF classifier on the words doesn't beat genre, decade and budget (AUC 0.58 against 0.58, and without
    franchise films 0.58 against 0.66).

### Emotional arcs: does the shape of a script's tone go with profit? (on main)

Every sentence of 834 non-franchise scripts is scored for tone by a sentiment model. Each film becomes a
100-point curve, summarised by two numbers: how much the tone swings (D1) and how much higher it ends than
it starts (D2). One pre-registered test follows, with budget, genre, decade and scene length controlled,
then checks across models, layers, settings and samples.

- Code: `analysis/arcs/`. Notebooks: `notebooks/arcs/`. Page: `site/arcs/index.html`. How to run it:
  [scripts-arcs.md](scripts-arcs.md).
- Findings:
  - Swing has no relationship with profit: ρ +0.035, p 0.32.
  - Ending lift is a near miss: ρ +0.078, p 0.028, against a pre-set cutoff of 0.025. Its direction holds
    in 12 of 13 variants but not with VADER.
  - Adding the arc doesn't improve out-of-sample prediction.
- In progress: a fourth sentiment model (ModernBERT) as a robustness check, on branch
  `scripts-arcs-modernbert`. It can't change the main result.

## Strand 3: how it was received (critics and audiences)

### Reviews exploration (Àngela, branch `reviews-exploration`, not merged)

Do money, critics and audiences agree? Every correlation is tested against a shuffle within genre ×
decade, on 2,095 clean 100k+ films up to 2017.

- Code and notebook: `analysis/reviews/`.
- Findings:
  - The multiple goes with IMDb rating at ρ 0.27, Metacritic 0.26 and Rotten Tomatoes 0.33. All are far
    above the genre-and-decade baseline, which is itself about 0.1.
  - The critics agree with each other far more: Metacritic and Rotten Tomatoes at 0.93.
  - Review tone (VADER) goes with the multiple at 0.13 and with IMDb rating at 0.22. Varied vocabulary
    goes with a lower multiple (−0.16).
  - A disagreement score, rating rank minus money rank, splits films into "liked beyond their money" and
    "earned beyond their liking". Their reviews differ in words like *tarantino* and *noir* against
    *scares* and *slasher*.
- Status: 50 hand labels pending. The branch also writes `disagreement_residual` per film for the network
  strand to join on.

The scripts and reviews notebook on main covers part of this too: review tone goes with the multiple at
ρ +0.184, net of budget and year.

## Bringing it together: what predicts profit? (on main)

Every block of predictors side by side, trained on films up to 2010 and tested on 2011 to 2017.

- Notebook: `notebooks/what_predicts_profit.ipynb`.
- Out-of-sample R²:

  | Model | R² |
  |---|---:|
  | Budget, year, genre | −0.010 |
  | + network | 0.202 |
  | All pre-release blocks | 0.244 |
  | + critic and audience scores | 0.312 |

  The network is the best block known before release. On the quarter of films that have a screenplay,
  script features add nothing beyond the network.
- Also in the notebook:
  - Director hit streaks: +0.175, against +0.090 when each director's films are shuffled.
  - PageRank adds nothing beyond plain experience (ρ −0.003).
  - Communities of companies that work together differ in profit: 3 to 4% of residual variance, against
    0.2% by chance.
  - Rotten Tomatoes goes with the multiple at ρ +0.25. There are 158 critically loved flops and 205 panned
    hits.

## The picture so far

Each of these is a finding from one analysis, with its checks. None is a cause.

- **Who made the film carries most of the signal known before release.** Studio and people track records
  predict profit on later films, and returns cluster by shared people and companies.
- **What the script says carries almost none once budget, genre and era are known.** The words differ only
  through franchise films, arc shapes don't separate hits from flops, and the ending lift is a near miss.
- **Money and liking agree, but only moderately** (ρ around 0.25 to 0.33). The films where they part ways
  have their own vocabulary.

## Open decisions and to-dos

- **The paper's one story.** The three strands support a single line: who makes a film predicts its return,
  its text doesn't, and audiences only partly reward what earns. The group needs to agree on it.
- **Human validation sheets**, which only people can fill (until then each one reads NOT YET VALIDATED):
  - Scattertext: `analysis/scripts_text/outputs/validation_*.csv`.
  - Arcs: `analysis/arcs/validation/`, plus `analysis/arcs/outputs/s12_back_to_text.csv`.
  - Reviews: the 50-review sample on `reviews-exploration`.
- **Arcs Step 11** (script arc against review tone) waits on one answer: does whole-review VADER count as the
  same instrument as sentence-level VADER on scripts?
- **Merges**: `reviews-exploration`, `initial_actor_network` and `g/hpc-access` (HPC instructions) are still
  on their own branches.
- **Known weak spots**: the franchise flag is rough, release month is parsed but not checked, reviews stop
  at 2017, and the 100k+ tier has few flops.

## Where everything is

| Path | What |
|---|---|
| `README.md` | Data status, how to get the data, the graph and profit sections |
| `AGENTS.md` | Setup and rules for coding agents |
| `docs/` | This overview, data sources, data structure, course brief, one run guide per text analysis |
| `data/bundles/` | Compressed data that comes with the repo |
| `scripts/imdb/`, `scripts/network/` | Data parsing, the movie graph, franchise flag, network features, ROI similarity |
| `scripts/screenplays/`, `scripts/reviews/` | Screenplay and review features (numbers only) |
| `analysis/network/` | ROI similarity results |
| `analysis/scripts_text/` | Scattertext analysis: config, log, steps, committed outputs |
| `analysis/arcs/` | Emotional arcs: config, log, steps, HPC job, committed outputs |
| `notebooks/` | `movie_graph`, `what_predicts_profit`, `scripts_and_reviews`, `scripts_text/`, `arcs/` |
| `site/scripts/`, `site/arcs/` | Static pages for the two text analyses, open in any browser |

| Branch | Owner | State |
|---|---|---|
| `main` | all | Graph, profit predictors, ROI similarity, Scattertext, arcs |
| `reviews-exploration` | Àngela | Critic vs audience vs money, not merged |
| `initial_actor_network` | Niklas | Actor network and interactive page, not merged |
| `g/hpc-access` | Gyula | HPC instructions, not merged |
| `scripts-arcs-modernbert` | Gyula | ModernBERT check for the arcs, in progress |
