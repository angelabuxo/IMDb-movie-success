# IMDb and Wikipedia: data structure

This is the data behind the working question in the [README](../README.md), as of 4 November 2026. The raw data isn't in this repo, and [data-sources.md](data-sources.md) lists where it comes from.

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

## Money, studios and scores

These come from the Wikipedia infoboxes and article text, through `scripts/imdb/parse_infobox.py`. Counts are out of 2,764 movies.

| Field | Movies |
|---|---:|
| Budget and gross both in USD | 2,567 |
| Studios | 2,661 |
| Distributors | 2,725 |
| Producers | 2,624 |
| Starring | 2,750 |
| Metacritic score | 2,431 |
| Rotten Tomatoes score, Wikipedia text | 1,983 |
| Rotten Tomatoes score, Wikipedia or Wikidata | 2,740 |
| Critical response section | 2,443 |

The clean sample drops streaming films, rentals-only or regional grosses, non-USD figures, budgets under $1,000 and 2026 releases. That leaves 2,446 movies, with a median gross of 3.4 times budget. 169 grossed less than their budget.

Clean movies by genre, counting a movie once for each genre it has:

| Genre | All years | Since 2000 | Since 2010 |
|---|---:|---:|---:|
| Drama | 1,093 | 794 | 465 |
| Action | 875 | 709 | 478 |
| Comedy | 805 | 601 | 341 |
| Adventure | 753 | 592 | 398 |
| Thriller | 484 | 375 | 215 |
| Crime | 466 | 322 | 182 |
| Sci-Fi | 301 | 236 | 150 |
| Romance | 285 | 214 | 103 |
| Horror | 280 | 223 | 139 |
| Mystery | 272 | 217 | 121 |

## Checks on those fields

- **Hand-read sample.** I read 50 random rows against the raw infobox and 49 were right. The error was a parser bug, now fixed. No budget or gross under $1,000 remains.
- **Wikipedia against Wikidata.** 77.9% of 1,788 budgets and 84.3% of 1,744 grosses agree within 5%. 34 budgets and 88 grosses differ by more than 50%, so these figures carry real uncertainty.
- **Rotten Tomatoes.** The two sources agree within 5 points on 98.1% of 1,967 movies.
- **First signal.** Log gross ÷ budget correlates about 0.24 with both IMDb rating and Metacritic. That's exploratory and comes from a hit-heavy sample.

## Screenplays

MovieSum has 2,200 rows covering 2,188 distinct IMDb IDs, with scenes, characters and dialogue marked up. 1,181 are in the 100k+ set and 1,095 of the clean movies have a usable script. The median script has 205,872 characters, 126 scenes, 874 dialogue lines and 55 character labels.

Screenplays were written before release, so a film's success can't have shaped them, unlike reviews and ratings.

## What's still open

- **Reviews can't be joined yet.** The Hugging Face copy has no movie ID. Stanford's original archive should have review URLs, but its servers weren't reachable from my machine, and the archived copy was rate-limited when I tried it.
- **The sample favours hits.** A 100,000-vote cut drops films few people saw. Articles for the 10k+ tier are still being downloaded, and their budget coverage isn't measured yet.
- **The franchise flag is noisy.** Wikidata's series property also holds lists, so it marks Inception as part of the BBC's 100 greatest films. The "follows" property is cleaner but misses first films in a series.
- **Screenplays need checking.** 12 IDs appear twice, character labels aren't cleaned, and I haven't read any scripts to confirm the markup or whether they're final versions.
- **Release month isn't validated.** It's parsed for 2,762 movies, but I haven't checked a sample.
- **Studio names aren't merged.** The clean sample has 1,296 distinct name strings, and some will be the same company.
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
