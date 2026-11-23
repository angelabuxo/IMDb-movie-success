# Local explorer

A Streamlit app that shows everything the notebooks show, lets you play with the data, and lets you open any data or output file to check it. It runs on your own machine only. Don't deploy it: `data/raw` holds licensed data and copyrighted script and review text.

From a fresh clone:

```bash
git clone https://github.com/angelabuxo/IMDb-movie-success.git && cd IMDb-movie-success && ./explorer/run.sh
```

From a clone you already have:

```bash
git pull && ./explorer/run.sh
```

Then open <http://localhost:8501>. [`run.sh`](run.sh) creates `.venv`, installs `requirements.txt` (again only when it changes), unpacks `data/bundles/` into `data/raw/` (again when a pull adds a bundle) through [`../setup.sh`](../setup.sh), and starts Streamlit on localhost only. Arguments go to Streamlit, so `./explorer/run.sh --server.port 8502` picks another port. The first run takes a few minutes and needs `zstd` (`brew install zstd` on a Mac). On Windows, run it from WSL or Git Bash rather than PowerShell directly; in WSL install it with `sudo apt install zstd`, then rerun the launcher. The launcher supports both Unix `.venv/bin` and Windows `.venv/Scripts` virtual-environment layouts.

The app reads `data/raw` (or `$IMDB_DATA_DIR`) and the committed outputs in `analysis/*/outputs` and `analysis/network`. It writes nothing.

To check every page before you open the app, run `./run_all.sh --check` from the repo root, or just the page check:

```bash
.venv/bin/python -m explorer.check_pages            # every page, every notebook step and view
.venv/bin/python -m explorer.check_pages --quick    # each page once
```

It runs each page headless on your data folder and lists crashes, errors, warnings and every input file a page couldn't find, with how to make it. It also reads every file the Data inspector lists. The report goes to `logs/check_pages.log`. On a fresh clone after `./setup.sh` you should see one missing file, `tmdb_compare.csv`, which needs the TMDB file from Kaggle.

## Logging

`explorer/log.py` writes to `logs/explorer.log`, which rotates at 2 MB. Each line is tagged with the page it came from. It records every page crash with its traceback (the app still shows it), every `st.error` and `st.warning` a page shows, every missing input file, and Python warnings from pandas, numpy and plotly. Set `EXPLORER_LOG_LEVEL=DEBUG` to also log how long each page took. When a page can't find a file, show the note with `missing(text)` from `explorer/common.py` instead of `st.info`, and say in the text how to make the file. That way it's logged and the check counts it.

## Pages

Text embeddings, Embedding maps, Embedding map checks, Profit attribution, Tags and genres and Tags: follow-ups are listed under "Archive" in the sidebar: they show early analyses that don't feed the paper, and their code is in [`archive/`](../archive/README.md).

| Page | What it shows | Where it comes from |
|---|---|---|
| Home | Data status and sample sizes | `data/raw/features/graph/nodes.csv` |
| Data pipeline | Each cleaning and processing step at a glance: a flow diagram from sources to analyses, what each step does, drops and can miss, the clean-rule funnel (checked against the clean flag in `nodes.csv`), the clean sample each analysis started from, and each analysis's own text-cleaning counts | `movies_features_10k.csv` and `nodes.csv` for the funnel; `analysis/*/outputs` summaries for the rest |
| Screenplay text | Every step of `notebooks/scripts_text/` | The step modules' `load()`, `tables()`, `figures()` and `reading()` in `analysis/scripts_text`, the same calls the notebooks and `site/scripts` make |
| Text embeddings | Reviews, screenplays and summaries against profit and revenue: what the text adds beyond budget, year, genre and franchise against its shuffle null, the checks (TF-IDF, later films, rating, OCR damage, other samples), the similarity network, the words along the direction, a film map and a nearest-films lookup | `archive/analysis/embeddings/outputs`, and the vectors and film table in `data/raw/features/text_vectors` (from the `embedding-vectors` bundle) for the map and lookup |
| Network results | The four network analyses as charts against their shuffle nulls; per-film network features against return; a film's credits and linked films; graph composition | `analysis/network/*.json`, `film_network_features.csv`, `nodes.csv`, `edges.csv` |
| Whole graph | The movie graph projected to films (linked by shared people or companies) or to people (linked by shared films), with filters, a layout, Louvain communities and colouring by return, year or genre. Click a node or drag a box around several to list them and open one | `nodes.csv`, `edges.csv` |
| Film-only network | The Q2 film projection: films linked by a director, top-billed actor, producer or studio. Zoom, hover, click or lasso films to see credits, budget, gross, return and displayed-network centralities | `nodes.csv`, `edges.csv` |
| Node explorer | Any movie, person or company: buttons to its IMDb, Wikipedia or Wikidata page, every field, every credit (click a row to move there), their films over time, frequent collaborators, the infobox names merged into a company, and a 1- or 2-hop network to click through | `nodes.csv`, `edges.csv`, `companies.csv`, `film_network_features.csv`, the feature tables |
| Screenplays | Any of the 2,188 MovieSum scripts, scene by scene: search, only the scenes with one character, the character network, the MovieSum summary and the raw markup with the parser's counts | `data/raw/scripts/moviesum/*.jsonl`, `analysis/scripts_text/outputs/s02_parser_report.csv` |
| Embedding maps | Each vector space (screenplays, summaries, plots, reviews, tags, genres, cast and crew, sentiment arcs) in 2D by PCA, Isomap, t-SNE or UMAP, or all four side by side, coloured by return, net return, genre or year. Pick a film to mark it and its 10 nearest films on the map and open it in the node explorer. Under the map, the scores for how well each method keeps the space | `data/raw/features/embedding_maps.csv` and `embedding_map_scores.csv`, written by `archive/notebooks/embedding_maps.ipynb` and shipped in the `embedding-vectors` bundle |
| Embedding map checks | Whether the embedding maps hold up: stability of t-SNE and UMAP across seeds and settings, how far the spaces agree on each film's nearest films, 10 k-means clusters of each space against profit (raw and net, with a shuffle null), sorting by release year, and the blind hand-check sheet with its score once judged. Shows the notebook's own reading | `archive/analysis/embedding_maps/*.csv` (committed) and the markdown of `archive/notebooks/embedding_map_checks.ipynb` |
| Profit playground | Filter by tier, year, genre and budget, pick any axes, see hits and flops | `nodes.csv` (movie rows) |
| Profit attribution | Optional. How a film's return splits between budget, year, genre, franchise, each person and company on it, and what's left: shares by role against a shuffle null, a table of people and companies with bootstrap intervals, one film as a dollar waterfall, three profit measures compared, and the checks. A sidebar slider sets the share of gross the studio keeps | `archive/analysis/attribution/outputs/` (no raw data needed) |
| Release calendar | Every film by year and month of first showing, one green or red square per month, for a chosen measure (adjusted return, gross ÷ budget, share under budget, budget, gross, rating, count), with year and month toggles, the same by month and by year, and the films in any square | `nodes.csv`, `wiki_infobox.csv`, `wiki_infobox_10k.csv` |
| Tags and genres | How much each film is each of the 1,084 Tag Genome tags, and its genre split in percent: one film, one tag, a genre across films, and the checks against IMDb's genres, the survey answers and the two scoring methods. The Profit view sets tags, genre shares and genre pairs against revenue and return. Click a film in any table for its overview, with buttons to its full page here, IMDb and Wikipedia | `tag_relevance.npy`, `genre_shares.csv` (built by `archive/scripts/reviews/tag_shares.py`), `archive/analysis/tags/tag_shares.json`, `archive/analysis/tags/tag_profit.*` (built by `archive/scripts/reviews/tag_profit.py`) |
| Tags: follow-ups | Five follow-ups to the tag and profit analysis: tags predicted from plot summaries, the usual genre mix of directors and studios, seen and liked against earned, critics against audience, and franchise and release month as controls | `archive/analysis/tags/plot_tags.*`, `archive/analysis/tags/tag_followups.json` (built by `archive/scripts/reviews/plot_tags.py` and `tag_followups.py`); needs no data folder |
| Data inspector | Any CSV, TSV, Parquet or JSON file: rows with a text filter, column types and missing values, one column's distribution, duplicate keys | `data/raw/features`, `features/graph`, every outputs folder |
| Checks | Automatic sanity checks (stale outputs, unique IDs, budget and gross, README counts, run-together names) and a button that runs `pytest` | Everything above |
| Second money source | Wikipedia's budget and gross against TMDB's: agreement by tier, how far apart the figures are, and (with the TMDB file on your machine) a film-by-film scatter and the largest disagreements | `analysis/checks/tmdb_compare.json`, `data/raw/features/tmdb_compare.csv` |

## Notes

- Every table sorts by column, filters by text and downloads as CSV. Every figure zooms and shows details on hover.
- Steps whose saved outputs need files in `private/` (gitignored, created by running the step on the data) show a notice instead of their numbers.
- The Screenplays page shows full script text from your local MovieSum files. Read it to check the parsing; never quote it into the paper or a commit.
- On the notebook pages, script and review excerpts stay hidden unless you turn on **Show private text (local only)** in the sidebar. They're read from files already on your machine and never written anywhere.
