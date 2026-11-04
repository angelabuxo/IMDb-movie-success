# IMDb and Wikipedia: data structure

This is a candidate dataset for the final project, as of 4 November 2026. It's separate from the founder-exits design in [design.md](design.md), and the raw data isn't in this repo.

The scope is non-adult movies with at least 100,000 IMDb votes. Wikipedia figures cover all 2,763 pages, pulled on 4 November 2026.

```text
PERSON  name.basics
key: nconst
columns: primaryName, birthYear, deathYear, primaryProfession, knownForTitles
in scope: 23,602 people, including 8,855 actors and 4,756 actresses
e.g. nm0000138 = Leonardo DiCaprio
   |
   |  CREDIT  title.principals
   |  columns: tconst, ordering, nconst, category, job, characters
   |  in scope: 61,536 credits, 22.3 per movie
   |  e.g. tt1375666 + nm0000138 + actor + ["Cobb"]
   v
MOVIE  title.basics + title.ratings
key: tconst
columns: primaryTitle, startYear, genres, numVotes, averageRating
in scope: 2,764 movies, 1921-2026, median 200,519 votes
e.g. tt1375666 = Inception, 2010, Adventure/Sci-Fi/Thriller, 2,886,189 votes, 8.8
   |
   +-- WIKIDATA  wiki/wd_tt_enwiki.tsv
   |   columns: imdb, item, article
   |   in scope: 2,762 movies linked, 2 without a link, 1 linked to two pages
   |             (Apocalypse Now and Apocalypse Now Redux)
   |   |
   |   v
   |   WIKIPEDIA ARTICLE  wiki/articles.jsonl
   |   columns: title_requested, title, tconst, pageid, revid, timestamp, wikitext
   |   in scope: 2,763 pages, median 46,446 characters and 188 distinct links
   |             2,756 have a plot-type section (median 4,166 characters)
   |   e.g. Inception, page 23270459, 142,336 characters, 457 distinct links
   |
   +.. REVIEW URL  urls_*.txt in the Stanford archive (unconfirmed)
       |
       v
       REVIEW  stanfordnlp/imdb on Hugging Face
       columns: text, label (no movie ID)
       in scope: 25,000 train + 25,000 test, labelled, half positive and half negative
                 50,000 unsupervised, no sentiment label
       median length: about 960-980 characters
```

## Networks this gives us

| Network | Size | Status |
|---|---|---|
| Movies sharing any person | 2,764 nodes, 168,791 edges (4.4% of possible pairs) | Counted, not built |
| Movies sharing a cast member (`actor`, `actress`, `self`) | 2,764 nodes, 67,331 edges (1.8% of possible pairs) | Counted, not built |
| Movies linked by Wikipedia | 19,108 links | Rough count from the wikitext |

## Wikipedia coverage

Links come from Wikidata entries that hold both an IMDb ID and an English article, so a page whose entry has no IMDb ID is missed.

| Minimum votes | Non-adult movies | English article | % |
|---:|---:|---:|---:|
| none | 750,810 | 150,126 | 20.0 |
| 100 | 147,331 | 92,926 | 63.1 |
| 1,000 | 49,431 | 43,872 | 88.8 |
| 10,000 | 12,673 | 12,562 | 99.1 |
| 100,000 | 2,764 | 2,762 | 99.9 |

Actors are covered less well. Of the 13,611 actors credited on movies with 100,000+ votes, 9,822 (72.2%) have an English article.

To test the misses, I checked 100 unlinked movies with 1,000+ votes against Wikipedia. Four had their own page and two more were covered on a shared page.

## What's still open

- **Reviews can't be joined yet.** The Hugging Face copy has no movie ID. Stanford's original archive should have review URLs, but its servers weren't reachable from my machine, and the archived copy was rate-limited when I tried it.
- **Credits are partial.** `title.principals` lists principal cast and crew, so people lower down the credits may be missing, along with the links they'd create.
- **Plot sections are unverified.** I found them by matching headings such as "Plot", "Synopsis" and "Premise", and haven't read a sample.
- **Wikipedia links are a rough count.** It's a simple text match that doesn't follow redirects, and the total isn't checked against the API.
- **Existing Wikidata links aren't checked.** I tested for missed pages, not whether the links we have are right.
- **Licences.** IMDb's data is for personal and non-commercial use, and I haven't read its full terms. Please don't commit the raw files.

## Sources and scripts

- IMDb files from <https://datasets.imdbws.com/>, snapshot of 4 November 2026.
- Wikidata through the QLever endpoint, at least as recent as 31 October 2026.
- English Wikipedia wikitext through the MediaWiki API, pulled 4 November 2026.
- Reviews from <https://huggingface.co/datasets/stanfordnlp/imdb>.

`scripts/imdb/` has the code behind these numbers. The scripts expect the data in a local folder with `raw/`, `wiki/` and `reviews/`, and they run from there. The SQL files use the DuckDB command line.
