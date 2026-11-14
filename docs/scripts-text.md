# Screenplay text analysis (Scattertext)

Do the screenplays of profitable and unprofitable films use different words? This is the screenplay NLP
part of the project: 1,095 clean films with 100,000+ IMDb votes and a MovieSum script, compared within
genre and decade, with a shuffle null, a second method, stability checks and franchise checks.

- Notebooks: [`notebooks/scripts_text/`](../notebooks/scripts_text/), `01_` to `12_`, one per step.
- Static page: [`site/scripts/index.html`](../site/scripts/index.html), built from the same saved outputs, so its
  numbers can't differ from the notebooks'. Open it in a browser; it needs no server or internet.
- Code: [`analysis/scripts_text/`](../analysis/scripts_text/). Each step is a module `sNN_*.py` with `run()`
  (compute and save), `load()` (read what was saved), tables, figures, a generated plain-language reading,
  and `SECTION`, the text that appears in both the notebook and the page.
- Parameters: [`analysis/scripts_text/config.yaml`](../analysis/scripts_text/config.yaml). Every choice is
  there, including the seed and the primary analysis, fixed before results were seen.
- Decisions, numbers per step and anomalies: [`analysis/scripts_text/LOG.md`](../analysis/scripts_text/LOG.md).
- Which parameters produced which output: `analysis/scripts_text/outputs/run_log.json` (config hash per step).

## Run it

Get the data first (see the README). The analysis reads `data/raw/features/movies_features.csv` and the
MovieSum JSONL files under `data/raw/scripts/moviesum/`. Set `IMDB_DATA_DIR` if the data lives elsewhere.

Environment (tested 11 November 2026, Python 3.13):

```bash
pip install pandas==3.0.5 numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 spacy==3.8.16 plotly==7.1.0 pyyaml==6.0.3 pyarrow==25.0.1 nbformat==5.11.1 nbconvert==7.17.1 ipykernel==7.3.0 mistune kaleido==1.5.1
```

```bash
pip install --no-deps scattertext==0.2.2 && python -m spacy download en_core_web_sm
```

scattertext 0.2.2 pins an older SciPy that has no wheel for Python 3.13, so it is installed without its
dependencies; everything it needs is in the first line.

Then, from the repo root:

```bash
python -m analysis.scripts_text.build_notebooks
```

```bash
python -m analysis.scripts_text.make_site
```

The first writes and executes the 12 notebooks (about 15 minutes; the spaCy pass in Step 3 is most of it,
then cached). The second builds the page. To run one step: `python -m analysis.scripts_text.build_notebooks 06`.

## The primary analysis

Fixed in `config.yaml` before results: definition (C), the bottom vs top quarter of log(gross/budget) cut
within genre × decade cells; the dialogue layer; character names removed; stopwords kept; no lemmatisation;
minimum document frequency 5; score log10((rate_successful + 0.1) / (rate_unsuccessful + 0.1)) with rates
per 10,000 words; threshold log10(2). Everything else is labelled exploratory.

## Outputs

`analysis/scripts_text/outputs/` (committed) holds aggregate tables only: per-film numbers, term rates and
scores, null and stability results, model results, and the two validation sheets with their empty columns.

`analysis/scripts_text/private/` (gitignored) holds anything with script text: the extracted layers, token
caches, document-term matrices, KWIC windows, validation previews and the Scattertext explorers. The
explorers are built from word counts only and are checked for runs of dialogue, but they stay local as
the brief asks. Notebook cells that show script text are tagged `private` and their outputs are cleared
before commit. The page is scanned for runs of dialogue before it is written and refuses to build if it
finds any.

## Human validation

Step 11 prepares two sheets: 25 random scripts and 30 random terms, with empty columns for people to fill.

1. Open `analysis/scripts_text/outputs/validation_scripts.csv` and `validation_terms.csv`.
2. Read the matching previews and KWIC lines in `analysis/scripts_text/private/`.
3. Fill the empty columns (never the agent).
4. Run `python -m analysis.scripts_text.s11_validation score`, then rebuild Step 11, Step 12 and the page.

Until then the notebooks and the page say NOT YET VALIDATED.

## Copyright

MovieSum is CC BY-NC 4.0 and the screenplays are under copyright. Never commit anything from `private/`,
full scripts or long excerpts. KWIC windows are at most 15 words and stay local.
