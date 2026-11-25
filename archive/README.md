# Archive

Early analyses from 11 to 16 November, before Niklas and Àngela took on more of the work. Nothing in the paper reads them. They're kept because they still run and the explorer still shows them under "Archive".

The folders keep their old layout under `archive/`, so `archive/analysis/attribution/` was `analysis/attribution/`. `analysis/__init__.py` and `explorer/__init__.py` add the archive folders to their package paths, so the old module names still work: `python -m analysis.attribution.run` and `from explorer import map_pages` run as before. Their tests are still in `tests/` and still run.

## What's here

**Profit attribution.** `analysis/attribution/`, `notebooks/profit_attribution.ipynb`, `docs/profit-attribution.md` and the explorer page `explorer/attribution_page.py`. It splits each film's return between its budget, year, genre and franchise, every person and company on it, and what's left. Cross-validated R² goes from 0.115 with budget, year and genre to 0.299 with franchise, people and companies, and 71.0% of the variance stays unexplained. It's a ridge model, which isn't a course method. To rerun it from the repo root (about 17 minutes, then a few seconds for the notebook):

```bash
python -m analysis.attribution.run
```

```bash
python -m analysis.attribution.build_notebook
```

**Text embeddings.** `analysis/embeddings/`, `notebooks/embeddings.ipynb` and `explorer/embedding_page.py`. Reviews, screenplays and summaries embedded with all-MiniLM-L6-v2 and set against return. The HPC scripts are in `analysis/embeddings/hpc/`.

**Embedding maps.** `analysis/embedding_maps/` (committed scores and checks), `analysis/film_spaces.py`, `analysis/projections.py`, `notebooks/embedding_maps.ipynb`, `notebooks/embedding_map_checks.ipynb` and `explorer/map_pages.py`. Every vector space in 2D by PCA, Isomap, t-SNE and UMAP.

**Tag Genome tags.** `analysis/tags/`, `scripts/reviews/` (`tag_shares.py`, `tag_profit.py`, `tag_followups.py`, `plot_tags.py` and the notebook builders) and `notebooks/tag_*.ipynb`. The explorer's "Tags and genres" and "Tags: follow-ups" pages are still in `explorer/pages.py`. The scripts run from the data folder, as before:

```bash
cd data/raw && ../../.venv/bin/python ../../archive/scripts/reviews/tag_shares.py
```

## Not here

The network-reach scripts (`scripts/network/pair_returns.py`, `team_history.py`, `movers.py`, `movie_assortativity.py`), `notebooks/movie_graph.ipynb`, `notebooks/network_reach.ipynb` and `notebooks/scripts_and_reviews.ipynb` are also early work outside the paper. They stayed where they were because the explorer's Network results page, the network tests and the TMDB robustness check still read them.

The sentiment-arc analysis was removed earlier and lives on the `archive/scripts-arcs` branch (see the main README).

The saved notebooks weren't rerun after the move. Their code cells now find the repo root two folders up, but their outputs are the ones from 16 November.

## Added in the cleanup of 25 November 2026

The repo moved to `data/`, `notebooks/`, `src/`, `paper/` and `archive/`. These came here because the paper no longer uses them:

- `analysis/scripts_text/` and `notebooks/scripts_text/`: the Scattertext word comparison.
- `notebooks/`: the long notebooks `q1_genre`, `movie_graph`, `network_reach`, `scripts_and_reviews`, `tmdb_second_source` and `what_predicts_profit`. `what_predicts_profit` still feeds a test.
- `docs/`: the course brief summary, the gap list, the overview PDFs and the README from before the cleanup.
- `site/`: the static Scattertext page.
