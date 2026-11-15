# What makes a movie highly profitable?

**Working question:** what is the best predictor of a movie being highly profitable, and does profit agree with whether people liked the film?

This is the final project for DTU 02805 Social Graphs and Interactions, autumn 2026, by Log-Log Legends: Gyula Kürthy, Àngela Buxó and Niklas Johansen. Course group repo: <https://github.com/horrrt/02805_social_graphs>

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

Adding movies with 10,000 to 99,999 votes brings the clean sample to 6,851, with 4,570 since 2000. That tier's parsing isn't hand-checked yet. (After the 15 November infobox fixes it's 6,843, with 4,567 since 2000: nine films with non-USD budgets or a Netflix release left and one wrongly excluded film joined; see [TESTING.md](TESTING.md).)

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
| `movie-graph-2026-11-15.tar.zst` | 12 MB | The graph and the infobox and movie tables after the 15 November parser fixes (see [TESTING.md](TESTING.md)). It replaces the same files from the two 11 November archives above |
| `extra-tables-2026-11-15.tar.zst` | 0.5 MB | Franchise flags, MovieSum IDs and screenplay statistics, and Tag Genome review counts per film, which no other archive holds |
| `extra-tables-2026-11-15.tar.zst` | 0.5 MB | Franchise flags, MovieSum IDs and stats, and Tag Genome review counts per film, which no other archive holds |

Unpack them from the repo root, oldest first, so a newer archive replaces the files it updates:

```bash
mkdir -p data/raw && for f in $(ls data/bundles/*.tar.zst | sed -E 's/.*-([0-9]{4}-[0-9]{2}-[0-9]{2})\.tar\.zst$/\1 &/' | sort | cut -d" " -f2); do tar --zstd -xf "$f" -C data/raw; done
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

## Explore and check everything locally

The explorer is a local app with every notebook step, the network results, the whole graph and every node in it, the screenplays scene by scene, a profit playground, an inspector for every data and output file, and automatic checks. To run it, clone the repo and start it with one command:

```bash
git clone https://github.com/angelabuxo/IMDb-movie-success.git && cd IMDb-movie-success && ./explorer/run.sh
```

Then open <http://localhost:8501>. The first run takes a few minutes: it creates `.venv`, installs `requirements.txt` and unpacks `data/bundles/` into `data/raw/`, so nothing needs downloading. Later runs start in seconds. You need a recent Python 3 (we use 3.14) and `zstd` for the unpacking (`brew install zstd` on a Mac).

If you already have a clone, pull and run:

```bash
git pull && ./explorer/run.sh
```

The script reinstalls the packages when `requirements.txt` changes. It unpacks the bundles only when `data/raw/` has no movie graph, so after a pull that adds a bundle, run the unpack command from [Get the data](#get-the-data) once. The app only listens on your own machine: `data/raw/` holds licensed data and copyrighted script text, so don't deploy it. [explorer/README.md](explorer/README.md) describes every page.

`python -m pytest` runs the tests; [TESTING.md](TESTING.md) covers what they test.

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
- [`docs/overview/overview.pdf`](docs/overview/overview.pdf): the whole project on one page, with [`project-details.pdf`](docs/overview/project-details.pdf) for a bit more on each part and [`literature.pdf`](docs/overview/literature.pdf) comparing our methods and findings with published research. LaTeX sources are next to them; rebuild with `tectonic -X compile overview.tex`.
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

`resolve_companies.py` gives each studio and distributor name a Wikidata ID, so "Miramax Films" and "Miramax" become one company while a label such as Magnet Releasing stays apart from its parent. It needs the Wikipedia and Wikidata APIs the first time (about five minutes) and caches the answers. Every name, the ID it got and why are in [`data/company_ids.csv`](data/company_ids.csv), so a merge can be checked or overridden. The other two steps take about 15 seconds and writes `nodes.csv`, `edges.csv` and `movie_graph.graphml` to `data/raw/features/graph/`. The docstring of [`scripts/network/build_graph.py`](scripts/network/build_graph.py) lists every field and its coverage. [`notebooks/movie_graph.ipynb`](notebooks/movie_graph.ipynb) loads it and shows projections, centrality, a collaborator network and a cast measure that uses only earlier films. The infobox parser used to run unlinked names together ("Madras TalkiesLyca Productions") and to drop unlinked names in lists; since 15 November 2026 it reads every list item and every `<br>`-separated name (see [TESTING.md](TESTING.md)). `resolve_companies.py` still splits any run-together name left over, but only when every piece is a Wikipedia article. `wiki_producers.py` adds the producers the Wikipedia infobox names and IMDb's principals file lacks (about 3,900 credits), matched to IMDb IDs only when the match is unambiguous; each edge's `origin` column says whether it came from IMDb or Wikipedia.

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

It uses the 6,843 clean movies in the graph, with return as ln(gross / budget). For each role it takes every person or company credited on at least two of those films and measures how much of the variance in return group membership accounts for (`similarity`, like an intraclass correlation, 0 for random films). As a second measure it correlates each film's return with the mean of the group's other films (`loo_r`). The null shuffles returns across films 1,000 times, which keeps every group's size and every overlap between groups. Results are in [`analysis/network/roi_similarity.json`](analysis/network/roi_similarity.json).

Every role clusters. After adjusting for ln(budget), release decade and genre, similarity runs from 0.09 for the top-three cast to 0.17 for producers, against a null whose 95th percentile is at most 0.02, and no shuffle reached the observed value for any role. The result holds when we drop franchise films (0.07 to 0.14), rank-transform returns so no single film dominates (0.08 to 0.16), keep only films from 1980 on (0.09 to 0.17) or keep only the 100k+ vote tier (0.04 to 0.14). The one role that weakens is distributor in the 100k+ tier, at 0.04 against 0.13 in the full sample, so distributors cluster mostly through the smaller films. Many of those had a limited or video-on-demand release and a tiny theatrical gross.

The roles overlap, since a director's films often share a studio, a composer and a cast. To see how much survives the studio, the `adjusted_within_studio` variant shuffles returns only among films of the same main studio. Directors then score 0.14 against a null of 0.06, and the other people roles look similar: roughly half their similarity is beyond what their studios share. We can't rank the roles against each other from this test.

What it doesn't show:
- It's descriptive. It uses every film at once, so it says returns cluster by group, not that a group's past predicts its next film. The `*_track` features in `film_features.py` test that with earlier films only.
- A shared data quirk clusters too. The highest returns include 1970s Hong Kong films listed with $100,000 budgets and $100 million grosses (Fist of Fury, The Way of the Dragon), and old Disney films whose gross includes decades of re-releases, so part of the top of the list is how those grosses were reported.
- Gross over budget isn't profit. Marketing is missing and cinemas keep part of the gross.

`features/roi_by_entity.csv` (local, not committed) lists each group's films, its mean return, and the mean shrunk towards zero by empirical Bayes, so a director with two lucky films doesn't top the list. Use it to look for patterns, then test any pattern you find the same way.

## How far the similarity reaches, and whether a record travels

Three scripts follow up on the shared-people result. They use the same 6,843 clean movies and the same adjusted return (ln(gross / budget) net of ln(budget), decade and genre). Run them from the data folder after the graph and `franchise_flag.py`:

```bash
../../.venv/bin/python ../../scripts/network/pair_returns.py
```

```bash
../../.venv/bin/python ../../scripts/network/team_history.py
```

```bash
../../.venv/bin/python ../../scripts/network/movers.py
```

The first takes several minutes and the other two under two. Each writes a JSON file next to [`roi_similarity.json`](analysis/network/roi_similarity.json), and `python scripts/network/network_figures.py` (from the repo root) draws [`analysis/network/figures/network_reach.pdf`](analysis/network/figures/network_reach.pdf) from them. [`notebooks/network_reach.ipynb`](notebooks/network_reach.ipynb) walks through all three with tables and plots. It reads only those JSON files, and `python scripts/network/build_notebook.py` rebuilds it. Every number below is without franchise films unless it says otherwise, and every null is 1,000 shuffles of returns.

### Film pairs: distance, shared people and scripts

[`pair_returns.py`](scripts/network/pair_returns.py) links two films when they share a key person (director, writer, producer, top-three cast, composer, cinematographer or editor) and measures the return correlation of film pairs, mean(z_i × z_j), which is 0 for random pairs.

- **Only direct links count.** Films one step apart correlate at +0.035 (+0.043 with franchise films), against a shuffle null whose 95th percentile is +0.004. Two steps apart it's +0.001, and further out it's zero or slightly negative. The small negatives are arithmetic: standardised returns sum to zero, so if linked pairs sit above zero the rest must sit a little below.
- **The more people two films share, the more alike their returns.** One shared person gives +0.027, two +0.086, three or four +0.153 and five or more +0.327 (641 pairs). With franchise films the top group reaches +0.465. Each step stays above a null that shuffles only within the film's main studio (95th percentiles +0.017, +0.046, +0.064 and +0.148), though the top group is small and the franchise flag is rough, so some of it is still sequels.
- **Companies and people both carry it.** Linked pairs with no studio or distributor in common still correlate at +0.021, and unlinked pairs that share a company at +0.032.
- **Similar scripts don't.** On the 1,689 films with a screenplay embedding (1,173 without franchise films), pairs that share a person correlate at +0.041. The same number of unlinked pairs with the most similar scripts correlate at +0.004, inside the null (95th percentile +0.012). Across deciles of script similarity, unlinked pairs stay between -0.009 and +0.006.
- **People do make somewhat similar films.** Within the same first genre, linked films' scripts have a cosine similarity of 0.664 against 0.648 for unlinked ones. That gap of 0.016 is about a sixth of the spread across all pairs (SD 0.093), and above a null that shuffles scripts within genre.

What it measures and misses: the pair correlation says returns cluster, using every year at once, so it isn't a prediction. Script similarity is the cosine between mean all-MiniLM-L6-v2 passage vectors, which captures shared topic and vocabulary and not quality or structure.

### Teams that have worked together before

[`team_history.py`](scripts/network/team_history.py) asks whether a film returns more when its key people have shared an earlier film. `repeat_share` is the share of pairs in the team who have, built from earlier years only over all 12,673 films in the graph.

- With franchise films, `repeat_share` goes with the adjusted return at ρ = +0.126, and adds 0.018 to out-of-time R² (fitted to 2010, tested on 2011 to 2017) over the team's own track record, experience and size.
- Most of that is sequels. Without franchise films it's ρ = +0.076 (within-year shuffle null: -0.017 to +0.028). Net of track record, experience and team size it's +0.022 (p = 0.11), and the out-of-time gain is 0.001.
- What's left sits after 2005: ρ = +0.135 (95% interval 0.095 to 0.171) for films from 2005 on, against +0.013 for 1990 to 2004 and -0.063 before 1990. We haven't tested why, and unflagged sequels are one candidate.

What it misses: "earlier film" means one with 10,000+ IMDb votes, and IMDb's principals file lists about ten people a title, so teams are partial.

### Movers: does a track record travel?

[`movers.py`](scripts/network/movers.py) takes every director, writer, producer and top-three cast credit where the person has at least two earlier clean films and the film's studios have at least five that the person didn't work on. A credit is a mover when none of the film's studios appears on any of the person's earlier films. It then regresses the film's adjusted return on the person's earlier record and the studio's, both standardised, with intervals that resample people.

| | Person's record | Studio's record |
|---|---:|---:|
| Movers (6,533 credits) | +0.045 (0.018 to 0.070) | +0.232 (0.199 to 0.262) |
| Stayers (6,772 credits) | +0.162 (0.124 to 0.200) | +0.188 (0.152 to 0.224) |

- **A person's record mostly stays behind.** At a studio that's new to them it predicts about a quarter of what it predicts for stayers, while the new studio's own record predicts at least as much as it does for stayers.
- **Directors keep the most.** Their record still gives +0.090 (0.020 to 0.160) after a move, and writers' +0.080 (0.020 to 0.139). Cast's is small but just above zero (+0.032, 0.003 to 0.062), and producers' (+0.027) can't be told from zero.
- **It isn't just noise in short records.** A person's record rests on a median of 5 films and a studio's on 21, which pulls the person's coefficient down. With both built from exactly three films, movers still give +0.063 for the person and +0.138 for the studio, and stayers +0.166 and +0.092.

What it doesn't show: moves aren't random. A studio hires someone for a project it has already chosen, so this isn't what would happen if people were reassigned. "Studio" is the production company in the Wikipedia infobox, and a film with several studios counts as a stay if any of them is familiar.

## Collection rules

We collect only through official APIs, bulk downloads, or the JSON endpoints a public page itself uses, and only where the terms do not prohibit it. Collection is gentle and recorded with its date. Sources whose terms forbid automated collection are listed as excluded and not used.
