# Screenplay emotional arcs

Once genre, decade, budget, franchise status and scene length are controlled, is the emotional shape of a
screenplay (how its tone rises and falls across the story) associated with the film's profitability? This is
the arc part of the screenplay work: 834 non-franchise films with 100,000+ IMDb votes and a MovieSum script
long enough for an arc, one pre-registered test, and the checks around it.

- Notebooks: [`notebooks/arcs/`](../notebooks/arcs/), `01_` to `13_`, one per step (Step 11 is not built yet,
  see below).
- Static page: `site/arcs/index.html`, built from the same saved outputs as the notebooks, so the numbers can't
  differ. It is built last, after the primary analysis, and opens in a browser with no server.
- Code: [`analysis/arcs/`](../analysis/arcs/). Each step is a module `sNN_*.py` with `run()` (compute and save),
  `load()` (read what was saved), tables, figures, a plain-language reading generated from the saved numbers,
  and `SECTION`, the text shared by the notebook and the page.
- Parameters: [`analysis/arcs/config.yaml`](../analysis/arcs/config.yaml). It was committed before any sentiment
  score or association was computed. Later amendments are dated in the file and explained in the LOG.
- Decisions, numbers per step and anomalies: [`analysis/arcs/LOG.md`](../analysis/arcs/LOG.md), newest last.
- Which parameters produced which output: `analysis/arcs/outputs/run_log.json` (config hash per step).

## Run it

Get the data first (see the README). The analysis reads `data/raw/features/movies_features.csv` and the
MovieSum JSONL files under `data/raw/scripts/moviesum/`. Set `IMDB_DATA_DIR` if the data lives elsewhere.

Environment (tested 14 November 2026, Python 3.13.16):

```bash
pip install pandas==3.0.5 numpy==2.5.3 scipy==1.18.1 scikit-learn==1.9.1 plotly==7.1.0 pyyaml==6.0.3 pyarrow==25.0.1 nbformat==5.11.1 nbconvert==7.17.1 ipykernel==7.3.0 mistune==3.3.4 vaderSentiment==3.3.2
```

Then, from the repo root, build and execute the notebooks (Steps 1 to 6 and 12 run in a few minutes; VADER
scores all 3.1 million sentences in under a minute):

```bash
python -m analysis.arcs.build_notebooks
```

One step only, for example the scene-order null:

```bash
python -m analysis.arcs.build_notebooks 06
```

The page, once Step 7 has run:

```bash
python -m analysis.arcs.make_site
```

Before that, a preview can be written anywhere outside `site/`:

```bash
python -m analysis.arcs.make_site --draft /tmp/arcs_preview.html
```

### The contextual models (DTU HPC)

DistilBERT SST-2 (pre-registered) and a three-class Twitter-RoBERTa (robustness) score every sentence and every
scene on a GPU. The job lives in [`analysis/arcs/hpc/`](../analysis/arcs/hpc/):

- `push_and_submit.sh` uploads the job and the private sentence table to an owner-only folder, smoke-tests both
  models and submits `score.lsf` (queue gpua40). `score_a10.lsf` is the same job for gpua10.
- `score_contextual.py` first times the models, then skips the full run if the projection is over 3 hours.
- `watch_and_fetch.sh` polls both copies, keeps whichever starts first, cancels the other, and copies the
  numeric results to `analysis/arcs/private/contextual/`.
- `score_stress.py` scores the constructed Step 3 stress sentences on the login node.

When the results are in, rerunning Step 3 brings them in, checks them, and writes
`outputs/s03_instrument_decision.json`. Steps 7 to 10 refuse to run until that file exists.

## The primary analysis

Fixed in `config.yaml` before any association was computed:
- Sample: not likely franchise and arc-ready (N = 834).
- Instrument: chosen by the fallback rule (DistilBERT SST-2 if its projected run is within 3 hours, otherwise
  VADER).
- Arc: the "both" layer (dialogue and description) at scene level, resampled to 100 points and smoothed over 11.
- Two descriptors: D1, the SD of the smoothed arc, and D2, the mean of the last 20% minus the mean of the first
  20%.
- Outcome: log(worldwide gross / budget).
- Controls: genre × decade cells (pooled below 5 films), log budget and mean sentences per scene, removed from
  both sides (Frisch-Waugh).
- Test: Spearman correlation of the residuals, with 1,000 permutations within cells, 2,000 bootstrap resamples
  and Bonferroni α = 0.025.

Everything else is labelled exploratory.

## Steps

| Step | Module | What it does |
|---|---|---|
| 1 | `s01_sample` | Sample funnel, outcome, controls |
| 2 | `s02_units` | Scenes and sentences, contraction fix, arc-ready filter |
| 3 | `s03_instruments` | VADER scores, contraction effect, stress tests, contextual intake and agreement |
| 4 | `s04_validation` | Blind human validation sheets and their scorer |
| 5 | `s05_arcs` | Resampling, smoothing, descriptors, proxy and split-half checks |
| 6 | `s06_shuffle` | Each script against 1,000 shuffles of its own scenes |
| 7 | `s07_primary` | The primary test |
| 8 | `s08_prediction` | Cross-validated gain in R² from the arc |
| 9 | `s09_robustness` | The primary statistic across instruments, layers, windows, grids, samples, outcome, genre rule, seeds |
| 10 | `s10_checks` | Franchise random-drop, transcript and leakage checks |
| 11 | (not built) | Script arc against audience reviews: waits for the group's decision on the instrument match |
| 12 | `s12_back_to_text` | The scenes and sentences behind the most extreme arcs, for a human check |
| 13 | `s13_interpretation` | What each analysis captures and misses, limitations, finding |

## Filling the human sheets

- `analysis/arcs/validation/scenes_sheet.csv`: label each scene from the private preview
  `analysis/arcs/private/validation/scenes_preview.html`. The preview shows no scores. Rater 2 works without
  seeing rater 1's labels. Then run `python -m analysis.arcs.s04_validation score`.
- `analysis/arcs/validation/scripts_sheet.csv`: the markup, draft or transcript, and ID-mapping checks, from
  `scripts_preview.html`.
- `analysis/arcs/outputs/s12_back_to_text.csv`: for each extreme, does the score match the text and what drives
  it, from `analysis/arcs/private/back_to_text/windows.html`.

Re-running a step never overwrites a filled human column.

## Outputs and what is never committed

`analysis/arcs/outputs/` (committed) holds numbers only: per-film descriptors, scene-level scores, arcs,
null and test results, and the sheets with their empty human columns. Films are identified only by the MovieSum
`movie_name`, with genre and decade as derived bins.

`analysis/arcs/private/` (gitignored) holds everything with script text or IMDb fields: the sentence table,
sentence-level scores, the HPC results as returned, the rater previews and the ≤ 15-word windows. Before
writing the page, `make_site.py` checks every 8-word run on it against every 8-word run of every script
sentence, and refuses to write if any match.
