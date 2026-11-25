# Testing lifecycle audit: 02805 movie-profit paper, 2026-11-15

Audited branch `g/pairwise-returns` at `39e704a`, the same code as `public/main`.

**Tier: coursework research pipeline.** The output is a six-page paper due 27 November 2026, plus notebooks and a static results site. There is no running service and no users, so most deploy, release and post-release practices are N/A. For research code, the important failures are *wrong numbers*, not crashes, so this audit maps each practice to its research equivalent where one exists.

**Summary:** the statistics are checked carefully: shuffle nulls, bootstrap, seed stability, franchise and transcript checks, human validation sheets, and a run log with config hashes. The code is not. There are zero automated tests, no CI, no linter, and nothing checks that the numbers in `paper/main.tex` match the current outputs.

## Pre-production (6/20 present, 9 missing, 5 N/A)

| Practice | Verdict | Evidence / reason | Priority |
|---|---|---|---|
| Unit tests | Present (2026-11-17) | 359 tests in `tests/`, running in about 30 s (`python -m pytest`; `-m "not slow"` takes about 12 s). They cover the functions behind every notebook and every page of the explorer, on synthetic data only: `test_stats_helpers.py` (bins, BH, permutations, Scattertext scores, shuffle null), `test_scripts_text.py` (name removal, cleaning, matching, TF-IDF), `test_network_and_parsing.py` (Wikipedia budget/gross parser, film projection, distance rings, pair correlation, within-stratum shuffles, screenplay features), `test_explorer.py` (130 tests: every explorer page under Streamlit's AppTest on a synthetic data folder, the full app navigation, all 12 steps of the screenplay-text notebook series, the four network result files, all four Whole graph modes with each colour, the node explorer on films, people and companies at one and two hops, the screenplay reader, and regressions for the explorer bugs below). `test_attribution.py` (20 tests for the optional profit attribution: the yearly CPI table and deflator, planted effects recovered, a film's own return kept out of its own split, the earlier-years version ignoring later films, the dollar split against brute force, the saved dollars adding up for every film, and the explorer page in each view). `pytest` is now in `requirements.txt` | - |
| Functional tests | Missing | no test runs a whole pipeline step (`run()`) on a small fixture. The step `run()` functions read `data/raw` and `private/` directly | Medium |
| Component tests | N/A | steps share files on disk, not service boundaries; functional tests cover this | - |
| Fuzz tests | N/A | parsers read fixed public dumps, not untrusted input | - |
| Property-based tests | Present (2026-11-15) | `tests/test_no_leakage.py` checks that `scripts/network/film_features.py` uses **only earlier years**: changing or adding a film never moves a same-year or earlier feature, and first-year films have zero history. It caught 4 of 4 hand-made leaks (same-year track, experience, `cast_reteamed`, network built before scoring) | - |
| Formal verification | N/A | no concurrency or safety-critical logic | - |
| Static analysis | Missing | no type checker or ruff config | Low |
| Mutation tests | Present (manual, 2026-11-15) | 16 hand-planted bugs across the leakage rule, pooling, BH, permutations, Step 7 null, TF-IDF, matching, arc maths, projection, screenplay network, money parser and sentence parser. The suite catches all 15 real ones; the 16th was equivalent to the original. Two more for the explorer (the cache key and the profit x = y fix) are both caught. Not automated | - |
| Coverage tests | Missing | add `pytest-cov` if useful. The report steps (`tables`, `figures`, `reading`) are deliberately untested | Low |
| Benchmark tests | N/A | the only benchmark timed the arcs sentiment models, which left main on 2026-11-17 (kept on `archive/scripts-arcs`) | - |
| Contract tests | Missing | steps hand CSVs to each other with no schema check. | Medium |
| Lint tests | Missing | no ruff, flake8 or pre-commit config | Low |
| Acceptance tests | Present | statistical acceptance: shuffle nulls (`analysis/scripts_text/s06_null.py`), franchise and leakage checks (`notebooks/scripts_text/10_franchise_leakage.ipynb`, `tests/test_no_leakage.py`) | - |
| Regression tests | Present (2026-11-16) | `tests/test_paper_numbers.py` recomputes every number in `paper/main.tex` and the hard-coded inputs of `paper/make_figures.py` from committed outputs, and fails on any number in the paper with no declared source. Reception numbers are read from the unmerged `reviews-exploration` branch and skipped without it | - |
| Smoke tests | Present (partial) | `analysis/*/build_notebooks.py` runs every notebook end to end with nbconvert's `ExecutePreprocessor`. It's manual and slow (2 h timeout), and there's no quick small-sample run | Medium |
| Usability tests | N/A | the paper gets read, not used. Peer reading counts as review, not testing | - |
| UI/UX tests | N/A | static results site only | - |
| Penetration tests | N/A | nothing exposed | - |
| Threat modeling | Present (data/licence) | `data/datasets.json` records each licence and forbids committing screenplays and reviews. `s11_validation.py` keeps text previews in `private/`, out of git | - |
| Performance tests | N/A | no heavy scoring runs left on main; the arcs HPC scripts are on `archive/scripts-arcs` | - |

Reproducibility (closest to static checks): **Present.** Fixed seeds are set in `analysis/scripts_text/config.yaml` and in each network and embedding script. `log_run()` and `config_hash()` in each `config.py` record the date, parameters and outputs of every step. `requirements.txt` pins every version.

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

- **Screenplay arcs removed on 2026-11-17.** `analysis/arcs`, `notebooks/arcs`, `site/arcs`, `tests/test_arcs.py` and the arc views in the explorer left main. The entries below that mention arcs describe that code, which is kept on the `archive/scripts-arcs` branch and the `archive/scripts-arcs-final` tag.

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
    - `archive/docs/overview/project-details.tex` and its rebuilt PDF.
  - The paper's other quoted numbers are unchanged: 0.09–0.17 vs at most 0.02, studio track ρ +0.27, and 0.090 of 0.175.
- **Found by the explorer's Checks page and fixed on 2026-11-15** (the second pipeline rerun; the old data is in `data/raw/_backup_before_list_and_currency_fix_2026-11-15/`):
  - **Currency codes written before `$` were read as US dollars.** Examples are `HUF$`, `NT$`, `S$`, `CAD$` and `GBP$`. *White God* is listed as "HUF$700 million (USD$2.5 million)" and was stored as a $700M budget; it now gets $2.5M.
    - `money()` now reads any `XX$` as currency XX. 20 budget or gross fields change.
    - 7 films leave the clean (USD) sample: *Videodrome*, *Scanners*, *The Brood*, *Shivers*, *Tideland*, *Creation* and *Gothic*.
    - `refresh_name_columns.py` now also copies budget, gross and their currency into `movies_features*.csv`.
  - **`names()` dropped unlinked names.** In a field with any wikilink, every unlinked name was dropped. In `{{Plainlist}}`, `{{ubl}}`, `{{ubil}}`, `{{idp}}` and `{{Indented plainlist}}` lists, items were dropped or merged. Names in `{{ill}}`, `{{nowrap}}` and `{{small}}` were dropped.
    - It now reads every list item: a line's link targets if it has links, otherwise its text. It also skips footnotes (`{{efn}}`) and nested templates.
    - On all 12,552 infoboxes this recovers about 28,000 names. About 1,700 names go, all of them merged runs, bracketed notes, link display texts kept next to their targets, footnote links or markup. A script checked each dropped name against those categories.
    - The "run-together" check now looks for names that split into two names found elsewhere. That count is 0 in both tiers. The 29 CamelCase names left (Columbia TriStar, Alexa PenaVega, Irish DreamTime) are real.
  - **The clean sample is now 6,843** (was 6,851). Nine films left: the seven above, *Videodrome*'s CAD budget counted among them, plus *Shaft* (2019), whose unlinked "Netflix (International)" distributor now triggers the streaming rule. *The Help* joined: a footnote saying it is "on Disney+ via Hulu" had wrongly matched that rule. The 100k+ count stays 2,446, and neither screenplay sample changes (*Videodrome* has no MovieSum script).
  - **Graph after the fix:** company nodes 6,958 → 12,654, Wikipedia producer credits about 2,000 → 3,909, edges 280,891 → 292,795.
  - **Network results hold.** The largest change is +0.043, in the movers' studio coefficients, which now rest on fuller studio lists. Two p-values cross 0.05:
    - Cast's record after a move is now +0.032 (0.003 to 0.062), just above zero.
    - Team history before 1989, net of the controls, crosses too.
  - **What was updated to match:**
    - README: every quoted network number and the clean-sample counts.
    - Paper: R² 0.196 / 0.237 / 0.309, base −0.013, studio track ρ +0.28, PageRank ρ +0.002, and Fig. 1 redrawn.
    - `project-details.tex` and its PDF.
  - **Desert Warrior's $150M budget against a $759,228 gross is real.** That's what its cited sources report. The check lists it as verified.
  - **Three arcs steps saved under an older `config.yaml`:**
    - `s05_arcs[vader]` and `s06_shuffle[vader]` were rerun. Their outputs are byte-identical, so only `run_log.json` changes.
    - `s03_compute_check` reads only `seed`, `instruments.compute_check_sentences`, `instruments.contextual` and `instruments.max_runtime_hours`, none of which changed. It's a timing benchmark that needs the private sentence table, so it's recorded as verified instead of rerun.
- **Clean rule tightened on 2026-11-15, after a hand-check of 50 random 10k-tier flops** (`analysis/checks/flop_sample_10k.csv`, seed 20261115):
  - **The check.** The parser matched the infobox in 50 of 50. Against the article text and Wikidata, 25 were clear worldwide flops, 6 used a US-only gross, 5 weren't really theatrical, 6 were pre-1975 rentals or similar and 8 were unclear.
  - **Rentals noted only in the citation were missed.** `scope()` reads the ref-stripped gross, but *The Haunting*'s ref says its figure is rentals. The new `gross_flags()` also looks for the whole word "rental(s)" in the raw value, refs included. It ignores URLs, ref names, the film's own title (*The Rental*, *Rental Family*) and Variety's 1995 headline "Worldwide rentals beat domestic take", which is cited for 1994 worldwide grosses. 75 infoboxes gain the flag across both tiers. Every match was read by hand: they are notes such as "figures are for rentals in US and Canada" or Variety's annual rental charts.
  - **Direct-to-video films.** `gross_flags()` adds `direct_to_video` when the article is in a direct-to-video category: 109 infoboxes.
  - **Pre-1975 films leave the sample.** Their grosses are mostly distributor rentals. Every copy of the rule now keeps 1975 to 2025 and drops `direct_to_video`: `build_graph.py`, `review_features.py`, `genome_stats.py`, `coverage_clean_sample.sql`, both `s01` steps (new `min_year_inclusive: 1975` in their `config.yaml`) and `scripts_and_reviews.ipynb`.
  - **Rerun:** `parse_infobox.py` (both tiers) → `refresh_name_columns.py` (only `gross_flag` changed: 5 rows at 100k+, 184 in the 10k table) → `build_graph.py` (edges byte-identical). Before the change the parser reproduced the bundled tables byte for byte.
  - **The clean sample is now 6,413** (was 6,843): 400 pre-1975 films, 15 with rentals and 15 direct-to-video left, none joined. Flops (gross under budget) 1,841 → 1,797, now 28.0%. The 100k+ tier is 2,352 with 167 flops (was 2,446 with 168).
  - **Not rerun yet:** `film_features.py` and the network steps after it, the `analysis/scripts_text` and `analysis/arcs` pipelines (their `expected` counts still say 2,446 and will show as mismatches), the notebooks, the README's network numbers and the paper. No new data bundle was made.
- **Explorer bugs found by a page-by-page sweep and fixed on 2026-11-15** (each has a test in `tests/test_explorer.py`):
  - **Profit playground crashed when x and y were the same column.** Plotly's trendline selected the column twice and narwhals raised a `DuplicateError`. The y list now leaves out the x column.
  - **The data inspector crashed on very wide files.** `screenplay_words.json` flattens to 2,068 rows × 162,424 columns, which Streamlit can't serialise. It now shows the first 500 columns with a warning.
  - **Clicks in the screenplay reader's Characters and Sentiment tabs didn't reach the Read tab.** The keyed selectbox ignored its new index, and the Read tab is drawn before the click is handled. The click is now handed over through session state with one rerun.
  - **Screenplay titles with underscores were garbled** (`Face (Off (1997)`); they now read `Face Off (1997)`.
  - **Screenplay text Step 1 (success groups) showed only errors on a machine that hadn't run it.** Its `load()` took the IMDb year, title and genres from `private/clean_table.parquet`; without that file every table and figure that used `year` failed. It now falls back to `movies_features.csv` in the data folder, the file Step 1 reads them from (`test_scripts_text.py`).
  - **Without the MovieSum files the explorer crashed** on the Screenplays page and on every film in the node explorer: the empty script index raised a `KeyError` instead of showing the "no MovieSum files" note.
  - **Cached loaders ignored file changes.** The modification time was passed as `_mt`, and Streamlit leaves arguments starting with an underscore out of the cache key, so a rebuilt graph or table was served from the cache until the explorer restarted.
- **Left as is:** about 9 names out of 150,000 come from one-off formatting (song-segment labels in *Moonwalker*, a bracket left open in *Layer Cake*).
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
