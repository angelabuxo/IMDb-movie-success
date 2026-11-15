# Testing lifecycle audit: 02805 movie-profit paper, 2026-11-15

Audited branch `g/pairwise-returns` at `39e704a`, the same code as `public/main`.

**Tier: coursework research pipeline.** The output is a six-page paper due 27 November 2026, plus notebooks and a static results site. There is no running service and no users, so most deploy, release and post-release practices are N/A. For research code, the important failures are *wrong numbers*, not crashes, so this audit maps each practice to its research equivalent where one exists.

**Summary:** the statistics are checked carefully: shuffle nulls, bootstrap, seed stability, franchise and transcript checks, human validation sheets, and a run log with config hashes. The code is not. There are zero automated tests, no CI, no linter, and nothing checks that the numbers in `paper/main.tex` match the current outputs.

## Pre-production (6/20 present, 9 missing, 5 N/A)

| Practice | Verdict | Evidence / reason | Priority |
|---|---|---|---|
| Unit tests | Present (2026-11-15) | 115 tests in `tests/`, running in about 11 s (`python -m pytest`; `-m "not slow"` takes about 3 s). They cover the functions behind every notebook, on synthetic data only: `test_stats_helpers.py` (bins, pooled cells, residualisation, r_min, BH, permutations, Scattertext scores, shuffle null), `test_arcs.py` (sentence parser, arc maths, scene-shuffle null, Step 7 association, Step 8 splits and OOF R²), `test_scripts_text.py` (name removal, cleaning, matching, TF-IDF), `test_network_and_parsing.py` (Wikipedia budget/gross parser, film projection, distance rings, pair correlation, within-stratum shuffles, screenplay features). `pytest` is now in `requirements.txt` | - |
| Functional tests | Missing | no test runs a whole pipeline step (`run()`) on a small fixture. The step `run()` functions read `data/raw` and `private/` directly | Medium |
| Component tests | N/A | steps share files on disk, not service boundaries; functional tests cover this | - |
| Fuzz tests | N/A | parsers read fixed public dumps, not untrusted input | - |
| Property-based tests | Present (2026-11-15) | `tests/test_no_leakage.py` checks that `scripts/network/film_features.py` uses **only earlier years**: changing or adding a film never moves a same-year or earlier feature, and first-year films have zero history. It caught 4 of 4 hand-made leaks (same-year track, experience, `cast_reteamed`, network built before scoring) | - |
| Formal verification | N/A | no concurrency or safety-critical logic | - |
| Static analysis | Missing | no type checker or ruff config | Low |
| Mutation tests | Present (manual, 2026-11-15) | 16 hand-planted bugs across the leakage rule, pooling, BH, permutations, Step 7 null, TF-IDF, matching, arc maths, projection, screenplay network, money parser and sentence parser. The suite catches all 15 real ones; the 16th was equivalent to the original. Not automated | - |
| Coverage tests | Missing | add `pytest-cov` if useful. The report steps (`tables`, `figures`, `reading`) are deliberately untested | Low |
| Benchmark tests | Present | `analysis/arcs/compute_check.py` benchmarks instruments and projects runtime before full scoring runs | - |
| Contract tests | Missing | steps hand CSVs to each other with no schema check. Only 9 `assert`s across all code (`analysis/arcs/s03_instruments.py` has 3) | Medium |
| Lint tests | Missing | no ruff, flake8 or pre-commit config | Low |
| Acceptance tests | Present | statistical acceptance: shuffle nulls (`analysis/arcs/s06_shuffle.py`, `analysis/scripts_text/s06_null.py`), robustness (`analysis/arcs/s09_robustness.py`), franchise and leakage checks (`analysis/arcs/s10_checks.py`, `notebooks/scripts_text/10_franchise_leakage.ipynb`) | - |
| Regression tests | Missing | no check that the paper's numbers still match the outputs after a rerun | **High** |
| Smoke tests | Present (partial) | `analysis/*/build_notebooks.py` runs every notebook end to end with nbconvert's `ExecutePreprocessor`. It's manual and slow (2 h timeout), and there's no quick small-sample run | Medium |
| Usability tests | N/A | the paper gets read, not used. Peer reading counts as review, not testing | - |
| UI/UX tests | N/A | static results site only | - |
| Penetration tests | N/A | nothing exposed | - |
| Threat modeling | Present (data/licence) | `data/datasets.json` records each licence and forbids committing screenplays and reviews. `s11_validation.py` keeps text previews in `private/`, out of git | - |
| Performance tests | Present | covered by `compute_check.py` and the HPC scripts in `analysis/arcs/hpc/` | - |

Reproducibility (closest to static checks): **Present.** Fixed seeds are set in `analysis/arcs/config.yaml` (`seed: 20261114`, plus 5 `stability_seeds`) and `analysis/scripts_text/config.yaml`, and 28 files use them. `log_run()` and `config_hash()` in each `config.py` record the date, parameters and outputs of every step. `requirements.txt` pins every version.

Gap: `scripts/get_data.py` downloads from `data/datasets.json` without checksums, so an upstream IMDb refresh silently changes the sample.

## Deploy (0/6, all N/A)

| Practice | Verdict | Evidence / reason |
|---|---|---|
| Integration tests | N/A | the closest equivalent is the full notebook rebuild, listed under Smoke tests |
| Tap compare | N/A | no old version serving traffic. The research equivalent is the paper-number regression check above |
| Load tests | N/A | no users |
| Shadowing | N/A | no traffic |
| Config tests | Missing (Low) | the `config.yaml` files are read without schema validation. A mistyped key falls back silently or crashes late |
| Soak tests | N/A | batch jobs only |

## Release (0/5, all N/A)

Canarying, monitoring, traffic shaping, feature flagging and exception tracking: **N/A**. Nothing is released to users. The paper is submitted once.

## Post-release (0/11, all N/A)

Teeing, profiling, logs/events, chaos, monitoring, A/B, tracing, dynamic exploration, real-user monitoring and on-call: **N/A**. There's no live system.
Auditing: **Present in research form.** `log_run()` writes a run log per step with config hashes, which shows which config produced which output.

## Bugs and quirks the tests found (2026-11-15)

- **Fixed and regenerated: `names()` in `scripts/imdb/parse_infobox.py` merged unlinked names separated by `<br>`** (`"Jane Doe<br>John Roe"` became `"Jane DoeJohn Roe"`), because `strip_code()` deleted the `<br>` tags before the split. The fix splits first, decodes HTML entities before splitting (so the `;` in `&nbsp;` isn't treated as a separator), and drops items that are only a parenthetical note. Checked on the real articles:
  - 100k+ tier: 48 of 2,763 infoboxes change, gaining 130 names in place of 41 merged ones.
  - 10k+ tier: 831 of 9,789 infoboxes change (561 producer, 279 studio, 63 distributor and 34 starring fields), gaining 2,629 names in place of 880 merged ones.
  - Every removed name was either a merged run of the new names or had a note such as "(uncredited)" attached. No real name is lost.

  **Pipeline rerun on 2026-11-15.** The steps were `parse_infobox.py` (both tiers) → `refresh_name_columns.py` (new: copies the name columns into `movies_features*.csv`, which nothing else rebuilds) → `resolve_companies.py` → `wiki_producers.py` → `build_graph.py` → `film_features.py` → `roi_similarity.py`, `pair_returns.py`, `team_history.py`, `movers.py` → `network_figures.py`, `build_notebook.py`, then `what_predicts_profit.ipynb` and `movie_graph.ipynb` were re-executed. `data/company_ids.csv` was regenerated: 7,298 → 7,677 names, with 334 merged names replaced by 713 split ones. The old files are in `data/raw/_backup_before_names_fix_2026-11-15/`.
  - **Graph:** 6,594 → 6,958 company nodes; 280,087 → 280,891 edges; Wikipedia-only producer credits now about 2,000.
  - **Results move only in the third decimal.** No p-value crosses 0.05. Headline studio similarity is 0.137 → 0.140, and the largest change is +0.021, in a small no-franchise subgroup.
  - **Updated to match:**
    - README: 11 numbers in the network sections, plus the two sentences about run-together names and the producer count.
    - `paper/main.tex`: R² 0.202 → 0.203, 0.244 → 0.245, 0.312 → 0.313, and PageRank ρ −0.003 → −0.002.
    - `docs/overview/project-details.tex` and its rebuilt PDF.
  - The paper's other quoted numbers are unchanged: 0.09–0.17 vs at most 0.02, studio track ρ +0.27, and 0.090 of 0.175.
- **Found by the explorer's Checks page, not fixed:**
  - **Two "clean" films have implausible budgets.** *White God* (2014) has $700M against a $0.58M gross, almost certainly 700 million Hungarian forints read as dollars. *Desert Warrior* is also flagged. Both sit in the clean sample.
  - **About 100 names still look run together** (24 in the 100k+ tier, 76 in the 10k+ tier). They come from cast and company lists written without any separator, such as `Alain DelonRenato Salvatori…`, which the `<br>` fix can't split.
  - **Three arcs steps were saved under an older `config.yaml`.** `s03_compute_check`, `s05_arcs[vader]` and `s06_shuffle[vader]` predate the last pre-registration amendments. Rerun them, or confirm those amendments can't affect them.
- **Not fixed: unlinked names inside templates such as `{{nowrap|...}}` are dropped by `strip_code()`.** This affects 1 field in the 100k+ tier and 24 in the 10k+ tier.
- **Quirk: the arcs sentence splitter drops a closing quote after end punctuation** (`'He said "Go."'` becomes `'He said "Go.'`). Words and sentiment are unchanged. A test pins the current behaviour.
- **Checked and fine: contraction joining.** MovieSum splits contractions as `does n't`, which the parser handles (148k matches). The `doesn 't` form, which it doesn't handle, occurs only 54 times in 24 scripts.
- **Note: on pandas 3, `s01_groups.quartile_labels` returns NaN, not `None`, for the middle half.** Callers use `isin(["low", "high"])`, so this is harmless.

## Top gaps to fix before 27 November

1. ~~**Test the no-leakage rule**~~ **Done 2026-11-15.** `tests/test_no_leakage.py` has 7 tests and runs in under a second. Run it with `python -m pytest tests/` (needs `igraph`, `numpy` and `pytest`). Set `FILM_FEATURES_PATH` to run it against another copy of the script. Not covered yet: the leakage checks in `analysis/arcs/` and `analysis/scripts_text/`.
2. **Check paper numbers against outputs** (High, ~1 h). Have `paper/make_figures.py` also write `paper/numbers.tex` with `\newcommand` macros from the outputs, use the macros in `main.tex`, and add a test that fails if a hard-coded number in `main.tex` disagrees with `outputs/`. This stops a rerun from leaving stale numbers in the paper.
3. **Add a quick smoke test on a small sample** (Medium, ~2 h). Add a `--sample 50` flag, or a fixture, so `pytest -m smoke` runs every step in minutes rather than hours, and run it before each merge to `main`.
4. **Check the files passed between steps** (Medium, ~1 h). Add an `assert_schema(df, columns, unique="tconst")` helper and call it where each step reads the previous step's CSV. Catch duplicate IMDb IDs, NaN budgets and broken joins at the point they happen.
5. **Pin the data** (Low, ~30 min). Add a `sha256` to each file in `data/datasets.json`, and have `get_data.py` warn when a download changes.

Optional: add a GitHub Actions workflow on the group repo that runs `ruff` and `pytest -m "not slow"` on pull requests. It costs about 20 minutes to set up, and teammates' PRs get checked automatically.
