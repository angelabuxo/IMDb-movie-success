# What makes a movie highly profitable?

**Working question:** what is the best predictor of a movie being highly profitable, and does profit agree with whether people liked the film?

This is the final project for DTU 02805 Social Graphs and Interactions, autumn 2026, by Log-Log Legends: Àngela Buxó, Gyula Kürthy and Niklas Johansen. Course group repo: <https://github.com/horrrt/02805_social_graphs>

## The plan

Networks: build a network of main actors, linked when they worked on a movie together in a chosen genre and period. Then look at how actors, directors, producers and studios relate to the most profitable movies.

Language: compare the sentiment of what's written about each movie with its profit. Does a huge profit mean people liked the film, or can the two disagree? That leads to the question of what success means here: money, popularity or positive reviews.

## Where the data stands (4 November 2026)

Counts are for the 2,764 non-adult movies with at least 100,000 IMDb votes.

| We need | Status | Count | Source |
|---|---|---:|---|
| Ratings | Have | 2,764 | IMDb |
| Main actors | Have | 3,084 actors in the top 3 billed, 5,709 in the top 5 | IMDb |
| Directors | Have | 2,762 | IMDb |
| Producers | Have | 2,624 | Wikipedia |
| Studios | Have | 2,661 | Wikipedia |
| Budget and gross in USD | Have | 2,567 with both | Wikipedia |
| Critic scores | Have | Metacritic 2,431, Rotten Tomatoes 2,740 | Wikipedia and Wikidata |
| Critic text | Have | 2,443 | Wikipedia |
| Screenplays | Have | 1,181 | MovieSum |
| Release month | Parsed, not validated | 2,762 | Wikipedia |
| Franchise flag | Rough only | 464 by the "follows" link | Wikidata |
| True profit | Not available | | Not published |
| Reviews linked to movies | Have, nothing after 2017 | 2,070 clean movies | MovieLens Tag Genome |
| Prior awards | Have | 1,479 of the 6,851 clean movies below appear | DLu/oscar_data |

After removing streaming films, rentals-only or regional grosses, non-USD figures and 2026 releases, 2,446 movies are left: 1,843 since 2000 and 1,093 since 2010. 1,095 of them have a usable screenplay.

Adding movies with 10,000 to 99,999 votes brings the clean sample to 6,851, with 4,570 since 2000. That tier's parsing isn't hand-checked yet.

## Decisions still open

- **Profit measure.** Neither option is real profit, because budgets leave out marketing and cinemas keep part of the gross. Gross ÷ budget is less affected by inflation and blockbuster size but favours micro-budget horror. Gross minus budget mostly rewards blockbusters.
- **Genre and dates.** See the genre table in [the data structure doc](docs/imdb-data-structure.md).
- **Which text carries the sentiment half.** MovieLens Tag Genome has audience reviews for 5,842 of the 6,851 clean movies, but none after 2017. Wikipedia's critic text covers later films too. Using both would let us compare critics with audiences, and would limit the audience half to 2017 or earlier.
- **Which sample to use.** The 100k+ tier is cleaner, and the 10k+ tier has far more flops.

## Known problems

- **The 100k+ sample favours hits.** Only 169 of the 2,446 movies grossed less than their budget, because that cut drops films few people saw. In the 10k to 100k tier, 38.1% did.
- **Franchises could swamp the result.** 11 of the 12 highest-margin films have a series or franchise link in Wikidata, and the current flag is noisy.
- **Network leakage.** Success can make actors more central, so a film's cast centrality should only use films released before it.

## Get the data

Everything the code reads lives in `data/raw/`, which git ignores. There are two ways to fill it.

### Data bundles (in the repo)

[`data/bundles/`](data/bundles/) holds compressed archives that come with every clone, so most of the data needs no download:

| Archive | Size | What it is |
|---|---:|---|
| `wikipedia-wikidata-2026-11-11.tar.zst` | 97 MB | Wikipedia article text for the 10,000+ vote movies (the hour-long pull), IMDb-to-Wikipedia links, Wikidata tables and the parsed infoboxes (budget, gross, studios, scores) |
| `moviesum-screenplays-2026-11-11.tar.zst` | 90 MB | The MovieSum screenplays with IMDb IDs |
| `screenplay-words-2026-11-11.tar.zst` | 10 MB | Dialogue word counts per screenplay |
| `text-features-2026-11-11.tar.zst` | 15 MB | Screenplay and review features and embeddings (numbers only) |
| `movie-graph-2026-11-11.tar.zst` | 9 MB | The movie graph: node and edge tables, GraphML, credits, people, the company IDs and the Wikipedia producer matches |

Unpack them from the repo root:

```bash
mkdir -p data/raw && for f in data/bundles/*.tar.zst; do tar --zstd -xf "$f" -C data/raw; done
```

[`data/bundles/README.md`](data/bundles/README.md) lists each archive's licence: Wikipedia text is CC BY-SA, and MovieSum and the Tag Genome are non-commercial. The same archives are also on the [data-2026-11-11 release](https://github.com/angelabuxo/IMDb-movie-success/releases/tag/data-2026-11-11).

### Everything else (download)

IMDb's raw files and the Tag Genome reviews aren't bundled: the IMDb files are 1.9 GB, and so are the reviews, over GitHub's file limit. The movie graph built from IMDb's files is bundled; rebuild it after downloading them only if you change the graph scripts (see [The movie graph](#the-movie-graph)). If you use a coding agent, ask it to set up the data, and it'll follow [AGENTS.md](AGENTS.md). To do it yourself:

```bash
python3 scripts/get_data.py --list
```

```bash
python3 scripts/get_data.py imdb genome
```

`--all` fetches every dataset (about 4.5 GB), and `python3 scripts/get_data.py wikipedia` redoes the Wikipedia pull (about an hour) if you need fresher articles than the bundle's.

## Scripts and reviews

[`notebooks/scripts_and_reviews.ipynb`](notebooks/scripts_and_reviews.ipynb) works through the group's notes on screenplays and reviews: hits against flops on character networks and text measures, sentiment arcs, distinctive words, script and review embeddings against profit, and how review sentiment lines up with ratings and money. Two scripts build its inputs. Run them from the data folder after downloading the data:

```bash
cd data/raw && ../../.venv/bin/python ../../scripts/screenplays/screenplay_features.py
```

```bash
cd data/raw && ../../.venv/bin/python ../../scripts/reviews/review_features.py
```

The first reads MovieSum and the second the Tag Genome reviews. Both write only numbers to `data/raw/features/`. The notebook also reads the screenplay embeddings in `features/embeddings_hpc/`.

## In this repo

- [`AGENTS.md`](AGENTS.md): setup steps and rules for coding agents.
- [`scripts/get_data.py`](scripts/get_data.py) and [`data/datasets.json`](data/datasets.json): the downloader and its list of files.
- [`docs/overview.md`](docs/overview.md): the whole project on one page, with [`docs/project-details.md`](docs/project-details.md) for a bit more on each part.
- [`docs/data-sources.md`](docs/data-sources.md): where each dataset comes from, with links and licences.
- [`docs/imdb-data-structure.md`](docs/imdb-data-structure.md): how the data fits together, with coverage and checks.
- [`docs/project-context.md`](docs/project-context.md): summary of the course's project brief.
- [`data/movie_links.csv`](data/movie_links.csv): IMDb, Wikipedia and Wikidata links for 12,673 movies.
- [`scripts/imdb/`](scripts/imdb/): the code behind the numbers.
- [`scripts/screenplays/`](scripts/screenplays/) and [`scripts/reviews/`](scripts/reviews/): screenplay and review features for the notebook.
- [`notebooks/`](notebooks/): analysis notebooks.

Raw data, article text and screenplays aren't in this repo. Please don't commit them.

## The movie graph

One network of the 12,673 movies with 10,000+ votes, the people who made them (actors with billing order, directors, writers, producers, composers, cinematographers, editors) and the companies that produced and distributed them. Build it from the data folder after downloading the IMDb files and parsing the infoboxes:

```bash
cd data/raw && mkdir -p features/graph && duckdb < ../../scripts/network/graph_tables.sql
```

```bash
../../.venv/bin/python ../../scripts/network/resolve_companies.py
```

```bash
../../.venv/bin/python ../../scripts/network/wiki_producers.py
```

```bash
../../.venv/bin/python ../../scripts/network/build_graph.py
```

`resolve_companies.py` gives each studio and distributor name a Wikidata ID, so "Miramax Films" and "Miramax" become one company while a label such as Magnet Releasing stays apart from its parent. It needs the Wikipedia and Wikidata APIs the first time (about five minutes) and caches the answers. Every name, the ID it got and why are in [`data/company_ids.csv`](data/company_ids.csv), so a merge can be checked or overridden. The other two steps take about 15 seconds and writes `nodes.csv`, `edges.csv` and `movie_graph.graphml` to `data/raw/features/graph/`. The docstring of [`scripts/network/build_graph.py`](scripts/network/build_graph.py) lists every field and its coverage. [`notebooks/movie_graph.ipynb`](notebooks/movie_graph.ipynb) loads it and shows projections, centrality, a collaborator network and a cast measure that uses only earlier films. About 1% of company names arrive run together from the infobox parser ("Madras TalkiesLyca Productions"); they are split only when every piece is a Wikipedia article, and otherwise stay unmerged. `wiki_producers.py` adds the producers the Wikipedia infobox names and IMDb's principals file lacks (about 1,800 credits), matched to IMDb IDs only when the match is unambiguous; each edge's `origin` column says whether it came from IMDb or Wikipedia.

## What predicts profit

[`notebooks/what_predicts_profit.ipynb`](notebooks/what_predicts_profit.ipynb) puts every predictor side by side: budget, year and genre; release month; franchise; network measures; critic scores; audience ratings and review sentiment. It trains on films up to 2010 and tests on 2011 to 2017. It also asks why a studio's track record predicts profit, whether network centrality adds to plain experience, whether alliances of companies differ in profit, and which films critics loved but lost money (and the reverse). Two more scripts build its inputs, after the graph:

```bash
../../.venv/bin/python ../../scripts/network/franchise_flag.py
```

```bash
../../.venv/bin/python ../../scripts/network/film_features.py
```

`franchise_flag.py` flags films in a series or franchise from Wikidata (follows, followed by, or part of a film series), which takes about 15 minutes the first time and is cached. `film_features.py` builds every network measure from films released before the one it describes, including PageRank in a collaboration network rebuilt for each release year, in about 10 seconds.

## Do films that share people have similar returns?

[`scripts/network/roi_similarity.py`](scripts/network/roi_similarity.py) asks whether films with the same director, writer, producer, top-three cast member, composer, cinematographer, editor, studio or distributor have more similar returns than chance. Run it from the data folder after the graph and `franchise_flag.py`. It takes about 15 seconds:

```bash
../../.venv/bin/python ../../scripts/network/roi_similarity.py
```

It uses the 6,851 clean movies in the graph, with return as ln(gross / budget). For each role it takes every person or company credited on at least two of those films and measures how much of the variance in return group membership accounts for (`similarity`, like an intraclass correlation, 0 for random films). As a second measure it correlates each film's return with the mean of the group's other films (`loo_r`). The null shuffles returns across films 1,000 times, which keeps every group's size and every overlap between groups. Results are in [`analysis/network/roi_similarity.json`](analysis/network/roi_similarity.json).

Every role clusters. After adjusting for ln(budget), release decade and genre, similarity runs from 0.09 for the top-three cast to 0.17 for producers, against a null whose 95th percentile is at most 0.02, and no shuffle reached the observed value for any role. The result holds when we drop franchise films (0.07 to 0.14), rank-transform returns so no single film dominates (0.08 to 0.16), keep only films from 1980 on (0.09 to 0.16) or keep only the 100k+ vote tier (0.03 to 0.14). The one role that weakens is distributor in the 100k+ tier, at 0.03 against 0.13 in the full sample, so distributors cluster mostly through the smaller films. Many of those had a limited or video-on-demand release and a tiny theatrical gross.

The roles overlap, since a director's films often share a studio, a composer and a cast. To see how much survives the studio, the `adjusted_within_studio` variant shuffles returns only among films of the same main studio. Directors then score 0.14 against a null of 0.07, and the other people roles look similar: roughly half their similarity is beyond what their studios share. We can't rank the roles against each other from this test.

What it doesn't show:
- It's descriptive. It uses every film at once, so it says returns cluster by group, not that a group's past predicts its next film. The `*_track` features in `film_features.py` test that with earlier films only.
- A shared data quirk clusters too. The highest returns include 1970s Hong Kong films listed with $100,000 budgets and $100 million grosses (Fist of Fury, The Way of the Dragon), and old Disney films whose gross includes decades of re-releases, so part of the top of the list is how those grosses were reported.
- Gross over budget isn't profit. Marketing is missing and cinemas keep part of the gross.

`features/roi_by_entity.csv` (local, not committed) lists each group's films, its mean return, and the mean shrunk towards zero by empirical Bayes, so a director with two lucky films doesn't top the list. Use it to look for patterns, then test any pattern you find the same way.

## Collection rules

We collect only through official APIs, bulk downloads, or the JSON endpoints a public page itself uses, and only where the terms do not prohibit it. Collection is gentle and recorded with its date. Sources whose terms forbid automated collection are listed as excluded and not used.
