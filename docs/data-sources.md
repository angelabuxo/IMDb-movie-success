# Data sources

Everything the project uses, with links, as of 4 November 2026. Per-movie links are in [`data/movie_links.csv`](../data/movie_links.csv).

| Source | What we use it for | Link | Licence or terms | Status |
|---|---|---|---|---|
| IMDb non-commercial datasets | Movies, ratings, cast and crew | [Docs](https://developer.imdb.com/non-commercial-datasets/), [files](https://datasets.imdbws.com/) | Personal and non-commercial use, full terms not read yet | Downloaded |
| Wikidata, through QLever | IMDb ID to Wikipedia links, second source for money, studios, scores and series | [QLever](https://qlever.dev/wikidata) | Wikidata's own terms not checked yet | Downloaded |
| English Wikipedia | Budget, gross, studios, producers, critic scores and critic text | [API](https://en.wikipedia.org/w/api.php) | Text is CC BY-SA, so credit it if we quote it Downloaded for 12,562 movies with 10,000+ votes |
| MovieSum | Screenplays and summaries with IMDb IDs | [Hugging Face](https://huggingface.co/datasets/rohitsaxena/MovieSum) | CC BY-NC 4.0, and the screenplays are still under copyright | Downloaded |
| Large Movie Review Dataset, Hugging Face copy | Review text and sentiment labels | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb) | Card licence listed as "other", not read yet | Downloaded, no movie IDs |
| Large Movie Review Dataset, URL lists | Links each review to a movie | [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/), [unofficial mirror](https://huggingface.co/datasets/hohai/aclImdb/tree/main/train) | Mirror has no licence or card | URL lists downloaded from the mirror, review text not yet matched to them |
| MovieLens Tag Genome 2021 | 2,624,608 IMDb reviews with IMDb IDs. Fields are review text and movie ID only, with no date or rating | [Page](https://grouplens.org/datasets/movielens/tag-genome-2021/) | CC BY-NC 3.0, cite the two papers in its readme. Reviews were collected from IMDb, method not stated | Downloaded, covers 5,842 of the 6,851 clean movies, nothing after 2017 |
| DLu/oscar_data | Oscar nominations and wins with IMDb film and person IDs | [GitHub](https://github.com/DLu/oscar_data) | Repo is BSD-2, but it was partly built from Oscars.org and scraped IMDb pages | Downloaded, pinned to commit `c5e9716`, 1,479 clean movies appear |
| TMDB movies on Kaggle | Third money source | [Kaggle](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) | Uploader lists Apache 2.0, but TMDB's API terms bar use "in connection with" a machine learning application | Not downloaded, and we shouldn't use it until the terms question is settled |

To cite MovieSum, use Saxena and Keller, "MovieSum: An Abstractive Summarization Dataset for Movie Screenplays", Findings of ACL 2024.

## Other sources worth using

None of these are in the project yet. "Tested" means I tried the source myself on 4 November 2026. The other details come from a read of each source's pages and haven't been repeated.

| Source | What it adds | Terms | How far I checked |
|---|---|---|---|
| [Wikipedia Pageviews API](https://doc.wikimedia.org/generated-data-platform/aqs/analytics-api/reference/page-views.html) | Attention before release, from July 2015. Older dumps go back to December 2007 | Reported CC0, no key needed | Tested on one movie |
| [Wikidata](https://www.wikidata.org/wiki/Property:P749) | Parent companies for merging studio names | CC0 | Not tested for this use |
| [BLS CPI API](https://www.bls.gov/developers/api_signature_v2.htm) | Inflation adjustment by year | Open, key optional | Not tested |
| [IMDB Spoiler Dataset](https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset) | Reported 573,913 reviews for 1,572 movies, with dates and ratings | Labelled CC BY 4.0, scraped | Not tested, Kaggle login likely |
| [Clapper Rotten Tomatoes](https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews) | Short critic quotes, scraped April 2023, reported about 1.4 million rows | Labelled CC0, which doesn't clear Rotten Tomatoes' terms or critics' copyright | Not tested, IDs need mapping through Wikidata |
| [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/) | 42,306 plots with revenue and actor data | CC BY-SA 3.0 US | Not tested |
| [OpusData](https://www.the-numbers.com/data-services) | Paid route to domestic, international and weekend box office. Offers reduced rates and one-time extracts | Price on request | Not tested, needs an email to sales |

Don't collect directly from these: IMDb pages, Rotten Tomatoes, Letterboxd, The Numbers and Box Office Mojo. Their terms or robots rules were reported to forbid it. YouTube statistics and Google Trends were also reported to have limits that rule them out.

No source has been found for marketing spend. Reviews for 2024 and 2025 aren't covered by anything above, and no open source of flops with IDs turned up beyond the 10k+ tier.

## How to get them

Use `scripts/get_data.py`, which reads its list of files from [`data/datasets.json`](../data/datasets.json). [AGENTS.md](../AGENTS.md) has the commands, so a coding agent can do this for you.

```bash
python3 scripts/get_data.py --list
```

OpusData and TMDB aren't included, for the reasons in the tables above.

## `data/movie_links.csv`

There's one row per movie and article, so 12,675 rows for 12,673 movies with 10,000+ votes.

| Column | Meaning |
|---|---|
| `imdb_id` | IMDb title ID, the key for joining everything |
| `imdb_url` | The movie's IMDb page |
| `wikipedia_url` | English Wikipedia article, blank if Wikidata has no link |
| `wikidata_url` | The movie's Wikidata entry, blank if it has no linked article |
| `min_votes_tier` | `100k` for 100,000+ votes, `10k` for 10,000 to 99,999 |
| `in_moviesum` | 1 if MovieSum has a screenplay for this ID |

12,562 of the movies have a Wikipedia link, and 2,020 are in MovieSum. Links come from Wikidata, and I opened only a few of them to test.
