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
| Reviews linked to movies | Partial, nothing after 2010 | 433 clean movies | Large Movie Review Dataset URL lists |

After removing streaming films, rentals-only or regional grosses, non-USD figures and 2026 releases, 2,446 movies are left: 1,843 since 2000 and 1,093 since 2010. 1,095 of them have a usable screenplay.

Adding movies with 10,000 to 99,999 votes brings the clean sample to 6,851, with 4,570 since 2000. That tier's parsing isn't hand-checked yet.

## Decisions still open

- **Profit measure.** Neither option is real profit, because budgets leave out marketing and cinemas keep part of the gross. Gross ÷ budget is less affected by inflation and blockbuster size but favours micro-budget horror. Gross minus budget mostly rewards blockbusters.
- **Genre and dates.** See the genre table in [the data structure doc](docs/imdb-data-structure.md).
- **Which text carries the sentiment half.** The linked reviews we have stop at 2010. Wikipedia's critic text, tested against critic scores, is the fallback, and MovieLens Tag Genome may fill the gap to about 2021 if its coverage holds up.
- **Which sample to use.** The 100k+ tier is cleaner, and the 10k+ tier has far more flops.

## Known problems

- **The 100k+ sample favours hits.** Only 169 of the 2,446 movies grossed less than their budget, because that cut drops films few people saw. In the 10k to 100k tier, 38.1% did.
- **Franchises could swamp the result.** 11 of the 12 highest-margin films have a series or franchise link in Wikidata, and the current flag is noisy.
- **Network leakage.** Success can make actors more central, so a film's cast centrality should only use films released before it.

## In this repo

- [`docs/data-sources.md`](docs/data-sources.md): where each dataset comes from, with links and licences.
- [`docs/imdb-data-structure.md`](docs/imdb-data-structure.md): how the data fits together, with coverage and checks.
- [`docs/project-context.md`](docs/project-context.md): summary of the course's project brief.
- [`data/movie_links.csv`](data/movie_links.csv): IMDb, Wikipedia and Wikidata links for 12,673 movies.
- [`scripts/imdb/`](scripts/imdb/): the code behind the numbers.

Raw data, article text and screenplays aren't in this repo. Please don't commit them.

## Collection rules

We collect only through official APIs, bulk downloads, or the JSON endpoints a public page itself uses, and only where the terms do not prohibit it. Collection is gentle and recorded with its date. Sources whose terms forbid automated collection are listed as excluded and not used.
