# Data sources

Everything the project uses, with links, as of 4 November 2026. Per-movie links are in [`data/movie_links.csv`](../data/movie_links.csv).

| Source | What we use it for | Link | Licence or terms | Status |
|---|---|---|---|---|
| IMDb non-commercial datasets | Movies, ratings, cast and crew | [Docs](https://developer.imdb.com/non-commercial-datasets/), [files](https://datasets.imdbws.com/) | Personal and non-commercial use, full terms not read yet | Downloaded |
| Wikidata, through QLever | IMDb ID to Wikipedia links, second source for money, studios, scores and series | [QLever](https://qlever.dev/wikidata) | Wikidata's own terms not checked yet | Downloaded |
| English Wikipedia | Budget, gross, studios, producers, critic scores and critic text | [API](https://en.wikipedia.org/w/api.php) | Text is CC BY-SA, so credit it if we quote it | 100k+ tier done, 10k+ tier in progress |
| MovieSum | Screenplays and summaries with IMDb IDs | [Hugging Face](https://huggingface.co/datasets/rohitsaxena/MovieSum) | CC BY-NC 4.0, and the screenplays are still under copyright | Downloaded |
| Large Movie Review Dataset, Hugging Face copy | Review text and sentiment labels | [Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb) | Card licence listed as "other", not read yet | Downloaded, no movie IDs |
| Large Movie Review Dataset, original | Should link reviews to movies through URL files | [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/), [archived file](https://web.archive.org/web/20261012055938id_/https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) | Not read yet | Not downloaded: Stanford unreachable from Gyula's machine, archive rate-limited |
| TMDB movies on Kaggle | Third money source, and more low-vote movies | [Kaggle](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) | Listed as Apache 2.0 by the uploader, TMDB's terms not checked | Not downloaded, needs a Kaggle login |

To cite MovieSum, use Saxena and Keller, "MovieSum: An Abstractive Summarization Dataset for Movie Screenplays", Findings of ACL 2024.

## How to get them

Run the commands from a local data folder outside the repo.

```bash
aria2c -x 16 -s 16 -k 1M -c https://datasets.imdbws.com/title.basics.tsv.gz
```

The other IMDb files follow the same pattern: `title.principals`, `title.crew`, `title.ratings`, `title.episode`, `title.akas` and `name.basics`.

```bash
aria2c -x 8 -s 8 -k 1M -c https://huggingface.co/datasets/rohitsaxena/MovieSum/resolve/main/train.jsonl
```

MovieSum also has `val.jsonl` and `test.jsonl`. TMDB has to be downloaded from the Kaggle page while logged in.

The Wikipedia pull is `scripts/imdb/pull_wiki.py` for the 100k+ tier and `scripts/imdb/pull_wiki_10k.py` for the 10k+ tier. Both can resume after a crash.

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
