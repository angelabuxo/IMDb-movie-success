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
| Critical response section (length only, via `parse_infobox.py`) | 2,443 |
| Reception / critical text (`features/wiki_reception.csv`, via `extract_reception.py`) | 2,485 of 2,763 pages |

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

## TMDB as a second money source

`scripts/imdb/tmdb_compare.py` compares the infobox budget and gross with TMDB's `budget` and `revenue` (Kaggle snapshot of 2026-11-15) for the 6,414 clean movies in `movies_features_10k.csv`. Every one has a TMDB row. The saved numbers are in `analysis/checks/tmdb_compare.json`. [`notebooks/tmdb_second_source.ipynb`](../notebooks/tmdb_second_source.ipynb) and the explorer's "Second money source" page both read that file.

| | Movies with a TMDB figure | Exactly equal | Within 10% | Within 25% | Off by 2x or more | Spearman (logs) |
|---|---:|---:|---:|---:|---:|---:|
| Budget, all clean | 6,015 | 71.9% | 83.2% | 92.3% | 2.5% | 0.985 |
| Gross, all clean | 6,132 | 13.1% | 85.4% | 90.3% | 4.6% | 0.987 |
| Budget, 100k+ | 2,348 | 70.1% | 85.5% | 95.4% | 1.1% | 0.992 |
| Gross, 100k+ | 2,346 | 9.6% | 92.7% | 95.5% | 1.7% | 0.995 |
| Budget, 10k to 100k | 3,667 | 73.0% | 81.8% | 90.3% | 3.5% | 0.975 |
| Gross, 10k to 100k | 3,786 | 15.3% | 80.9% | 87.1% | 6.4% | 0.970 |

On the 5,891 movies where TMDB has both figures, log(gross ÷ budget) correlates at 0.959 between the sources, and they make the same under-budget call for 96.1% (Wikipedia 1,533 flops, TMDB 1,632). In the 100k+ tier it's 98.5% of 2,343; in the 10k to 100k tier it's 94.5% of 3,548.

Three things limit what this shows:

- **The sources aren't independent.** 71.9% of TMDB's budgets are exactly Wikipedia's number, and 63.7% of its grosses are within 0.5%. The agreement mostly shows the parser read Wikipedia correctly, not that Wikipedia is right.
- **231 films change their under-budget label.** 97 are near break-even (Wikipedia gross ÷ budget between 0.8 and 1.25) and 134 aren't. 165 are flops only on TMDB and 66 only on Wikipedia. `scripts/imdb/tmdb_robustness.py` reruns two headlines with TMDB money in place of Wikipedia's (outcome, budget control and hit/flop cut-offs), on the 5,891 clean films where TMDB has both figures, after first reproducing each committed number exactly. Nothing changes direction: degree centralisation ρ -0.098 with Wikipedia money and -0.100 with TMDB's, review sentiment +0.187 and +0.185. The swap moves 7.6% of films between hit, middle and flop. The network results aren't covered, because their predictors are built from earlier films' Wikipedia returns.
- **Only clean films are tested.** A parser miss usually drops a film from the sample. Of the 6,259 films outside it, 4,230 have no parsed budget or gross, and TMDB has both figures for 739 of those. For 656 of them (45 in the 100k+ tier) the infobox has no field for the missing figure; a plain search of the article wikitext, independent of the parser, finds a non-empty line in 1 of 656. 73 have a raw value the parser didn't turn into a number, and 8 have no article linked. A seeded random 30 of the 73, labelled by the coding agent from their raw values (`analysis/checks/tmdb_unparsed_labels.csv`), held 24 non-dollar amounts the parser can't read (currency templates, currency names, a leaked `link=yes`), 4 with no money figure (French admissions, "Debated", a number with no currency) and 2 missed dollar figures (*Mr. Nice Guy*'s "$>25 million", a lower bound, and a gross computed by a conversion template). So parser misses barely touch the clean sample. Filling the 656 from TMDB would add films, mostly to the flop-heavy tier, which is a decision for the group.

Two assumptions: TMDB has no currency field, so its figures are read as US dollars, and a TMDB 0 counts as "no figure". TMDB is edited by users without citations, so a disagreement doesn't say which side is wrong.

I read the raw infobox value for nine of the largest disagreements and the parser had read all nine correctly. They were picked because they disagreed most, so that's not an error rate. TMDB had 4 revenues under $1,000 (a figure typed in millions), and where gross is off by 2x or more TMDB is the lower one 252 times against 29. My reading of the cases I looked at is that TMDB often holds the US-only gross for foreign films (*Samba*, *Microcosmos*), but I haven't checked that across the 252.

**One article per film.** Two Wikidata items can carry the same IMDb ID, so two films were linked to two articles: *Horizon: An American Saga – Chapter 1* (also the franchise article) and *Apocalypse Now* (also *Redux*). Readers that kept the first row got the franchise article for *Horizon*, which has a budget for two films and no gross, so it dropped out. `parse_infobox.py` now writes one row per film, keeping the article whose title matches the IMDb title (`analysis/wiki_links.py`). *Horizon* is back with its own figures and *Apocalypse Now* is unchanged; the `one-article-per-film-2026-11-16` bundle holds the fixed tables.

## What's still open

- **Reviews stop at 2017.** MovieLens Tag Genome covers 5,842 of the 6,851 clean movies with 1,621,933 reviews (median 163 per film), and 1,524 of those movies grossed under budget. Coverage is close to complete through the 2000s, 1,618 of 2,032 for the 2010s, and zero for 2018 onwards. Its reviews have no date or rating, and I haven't read a sample. The Large Movie Review Dataset is much smaller (1,376 clean movies, 20,514 reviews) and has almost nothing from 2010 onwards, but it does have sentiment labels.
- **The 100k+ sample favours hits.** Adding movies with 10,000 to 99,999 votes gives 6,851 clean movies, and 38.1% of those added movies grossed under budget, against 6.9% at 100k+. About half of that tier has budget and gross in USD, and its parsing isn't hand-checked. White God's $700 million budget is probably a misread currency.
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
