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

Then open <http://localhost:8501>. [`run.sh`](run.sh) creates `.venv`, installs `requirements.txt` (again only when it changes), unpacks `data/bundles/` into `data/raw/` (again when a pull adds a bundle) through [`../setup.sh`](../setup.sh), and starts Streamlit on localhost only. Arguments go to Streamlit, so `./explorer/run.sh --server.port 8502` picks another port. The first run takes a few minutes and needs `zstd` (`brew install zstd` on a Mac).

The app reads `data/raw` (or `$IMDB_DATA_DIR`) and the committed outputs in `analysis/*/outputs` and `analysis/network`. It writes nothing.

## Pages

| Page | What it shows | Where it comes from |
|---|---|---|
| Home | Data status and sample sizes | `data/raw/features/graph/nodes.csv` |
| Screenplay arcs | Every step of `notebooks/arcs/` | The step modules' `load()`, `tables()`, `figures()` and `reading()`, the same calls the notebooks and `site/arcs` make |
| Screenplay text | Every step of `notebooks/scripts_text/` | Same as above, from `analysis/scripts_text` |
| Network results | The four network analyses as charts against their shuffle nulls; per-film network features against return; a film's credits and linked films; graph composition | `analysis/network/*.json`, `film_network_features.csv`, `nodes.csv`, `edges.csv` |
| Whole graph | The movie graph projected to films (linked by shared people or companies) or to people (linked by shared films), with filters, a layout, Louvain communities and colouring by return, year or genre. Click a node or drag a box around several to list them and open one | `nodes.csv`, `edges.csv` |
| Node explorer | Any movie, person or company: every field, every credit (click a row to move there), their films over time, frequent collaborators, the infobox names merged into a company, and a 1- or 2-hop network to click through | `nodes.csv`, `edges.csv`, `companies.csv`, `film_network_features.csv`, the feature tables |
| Screenplays | Any of the 2,188 MovieSum scripts, scene by scene: search, only the scenes with one character, the character network, the per-scene sentiment from arcs Step 3 (click to jump to a scene), the MovieSum summary and the raw markup with the parser's counts | `data/raw/scripts/moviesum/*.jsonl`, `analysis/arcs/outputs/s03_scene_vader.csv.gz` |
| Profit playground | Filter by tier, year, genre and budget, pick any axes, see hits and flops | `nodes.csv` (movie rows) |
| Profit attribution | Optional. How a film's return splits between budget, year, genre, franchise, each person and company on it, and what's left: shares by role against a shuffle null, a table of people and companies with bootstrap intervals, one film as a dollar waterfall, three profit measures compared, and the checks. A sidebar slider sets the share of gross the studio keeps | `analysis/attribution/outputs/` (no raw data needed) |
| Data inspector | Any CSV, TSV, Parquet or JSON file: rows with a text filter, column types and missing values, one column's distribution, duplicate keys | `data/raw/features`, `features/graph`, every outputs folder |
| Checks | Automatic sanity checks (stale outputs, unique IDs, budget and gross, README counts, run-together names) and a button that runs `pytest` | Everything above |

## Notes

- Every table sorts by column, filters by text and downloads as CSV. Every figure zooms and shows details on hover.
- Steps whose saved outputs need files in `private/` (gitignored, created by running the step on the data) show a notice instead of their numbers.
- The Screenplays page shows full script text from your local MovieSum files. Read it to check the parsing; never quote it into the paper or a commit.
- On the notebook pages, script and review excerpts stay hidden unless you turn on **Show private text (local only)** in the sidebar. They're read from files already on your machine and never written anywhere.
