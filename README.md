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

The data isn't in git. If you use a coding agent, ask it to set up the data, and it'll follow [AGENTS.md](AGENTS.md). To do it yourself:

```bash
python3 scripts/get_data.py --list
```

```bash
python3 scripts/get_data.py --all
```

`--all` needs about 4.5 GB. Wikipedia articles are a separate step of about an hour: `python3 scripts/get_data.py wikipedia`. Everything goes in `data/raw/`, which git ignores.

To skip the Wikipedia pull, download the [data release](https://github.com/angelabuxo/IMDb-movie-success/releases/tag/data-2026-11-11) instead. It has the Wikipedia articles, Wikidata tables and parsed infoboxes (92 MB), and the screenplay and review features with their embeddings (15 MB). IMDb's files aren't in it, because IMDb's terms don't allow republishing, so still fetch those with `get_data.py imdb`:

```bash
gh release download data-2026-11-11 -R angelabuxo/IMDb-movie-success -D data/raw
```

```bash
cd data/raw && for f in *.tar.zst; do tar --zstd -xf "$f"; done
```

Each archive has a README with its file list and licences. The Wikipedia part is CC BY-SA, and the features are non-commercial because MovieSum and the Tag Genome are.

## In this repo

- [`AGENTS.md`](AGENTS.md): setup steps and rules for coding agents.
- [`scripts/get_data.py`](scripts/get_data.py) and [`data/datasets.json`](data/datasets.json): the downloader and its list of files.
- [`docs/data-sources.md`](docs/data-sources.md): where each dataset comes from, with links and licences.
- [`docs/imdb-data-structure.md`](docs/imdb-data-structure.md): how the data fits together, with coverage and checks.
- [`docs/project-context.md`](docs/project-context.md): summary of the course's project brief.
- [`data/movie_links.csv`](data/movie_links.csv): IMDb, Wikipedia and Wikidata links for 12,673 movies.
- [`scripts/imdb/`](scripts/imdb/): the code behind the numbers.

Raw data, article text and screenplays aren't in this repo. Please don't commit them.

## The movie graph

One network of the 12,673 movies with 10,000+ votes, the people who made them (actors with billing order, directors, writers, producers, composers, cinematographers, editors) and their studios and distributors. Build it from the data folder after downloading the IMDb files and parsing the infoboxes:

```bash
cd data/raw && mkdir -p features/graph && duckdb < ../../scripts/network/graph_tables.sql
```

```bash
../../.venv/bin/python ../../scripts/network/build_graph.py
```

That takes about 15 seconds and writes `nodes.csv`, `edges.csv` and `movie_graph.graphml` to `data/raw/features/graph/`. The docstring of [`scripts/network/build_graph.py`](scripts/network/build_graph.py) lists every field and its coverage. [`notebooks/movie_graph.ipynb`](notebooks/movie_graph.ipynb) loads it and shows projections, centrality, a collaborator network and a cast measure that uses only earlier films. Studio names aren't merged yet.

## Collection rules

We collect only through official APIs, bulk downloads, or the JSON endpoints a public page itself uses, and only where the terms do not prohibit it. Collection is gentle and recorded with its date. Sources whose terms forbid automated collection are listed as excluded and not used.
