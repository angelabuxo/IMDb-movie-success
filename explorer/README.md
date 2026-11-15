# Local explorer

A Streamlit app that shows everything the notebooks show, lets you play with the data, and lets you open any data or output file to check it. It runs on your own machine only. Don't deploy it: `data/raw` holds licensed data and copyrighted script and review text.

```bash
pip install -r requirements.txt          # streamlit, statsmodels and the analysis packages
streamlit run explorer/app.py            # then open http://localhost:8501
```

It reads `data/raw` (or `$IMDB_DATA_DIR`) and the committed outputs in `analysis/*/outputs` and `analysis/network`. It writes nothing.

## Pages

| Page | What it shows | Where it comes from |
|---|---|---|
| Home | Data status and sample sizes | `data/raw/features/graph/nodes.csv` |
| Screenplay arcs | Every step of `notebooks/arcs/` | The step modules' `load()`, `tables()`, `figures()` and `reading()`, the same calls the notebooks and `site/arcs` make |
| Screenplay text | Every step of `notebooks/scripts_text/` | Same as above, from `analysis/scripts_text` |
| Network | The four network analyses as charts against their shuffle nulls; per-film network features against return; a film's credits and linked films; graph composition | `analysis/network/*.json`, `film_network_features.csv`, `nodes.csv`, `edges.csv` |
| Profit playground | Filter by tier, year, genre and budget, pick any axes, see hits and flops | `nodes.csv` (movie rows) |
| Data inspector | Any CSV, TSV, Parquet or JSON file: rows with a text filter, column types and missing values, one column's distribution, duplicate keys | `data/raw/features`, `features/graph`, every outputs folder |
| Checks | Automatic sanity checks (stale outputs, unique IDs, budget and gross, README counts, run-together names) and a button that runs `pytest` | Everything above |

## Notes

- Every table sorts by column, filters by text and downloads as CSV. Every figure zooms and shows details on hover.
- Steps whose saved outputs need files in `private/` (gitignored, created by running the step on the data) show a notice instead of their numbers.
- Script and review excerpts stay hidden unless you turn on **Show private text (local only)** in the sidebar. They're read from files already on your machine and never written anywhere.
