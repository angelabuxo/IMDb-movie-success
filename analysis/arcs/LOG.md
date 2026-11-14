# Screenplay emotional arcs: decision and run log

Newest last. Each step records its numbers, anomalies, decisions and open questions.

## 2026-11-14 · Step 0: reproduce, reconcile, pre-register

**Where the inputs are.** `analysis/feasibility/FEASIBILITY.md` and `feasibility_numbers.json`, named in the
brief, are not on any branch (local or `origin`) or anywhere on this machine. The gate numbers were
therefore checked against the values stated in the brief. The earlier branch `scripts-scattertext` is merged
into `main` (PR #8); its LOG and config were read from `analysis/scripts_text/`.

**Reproduction (counts only, no sentiment computed).** Clean rule as in `scripts/imdb/coverage_clean_sample.sql`,
MovieSum IDs with ≥ 1 `<scene>` and ≥ 1 `<dialogue>`, longest script for the 12 duplicate IDs, the 8 scripts
with < 500 dialogue words excluded.

| Check | Expected | Reproduced |
|---|---:|---:|
| Clean 100k+ movies | 2,446 | 2,446 |
| … with a usable script | 1,095 | 1,095 |
| … after the < 500 dialogue-word exclusion | 1,087 | 1,087 |
| … not likely franchise (follows 146, named 133, either 219) | 868 | 868 |
| … arc-ready (≥ 20 scene tags and ≥ 300 dialogue+action sentences) | 834 | 834 |
| Script and Tag Genome reviews | 946 | 946 |
| … not likely franchise | 759 | 759 |

834 depends on the sentence rule. It reproduces with the regex now frozen in `config.yaml` (split after
. ! ? inside each dialogue / scene-description tag). Counting non-empty scenes instead of scene tags gives 833.
The latest film with Tag Genome reviews is from 2017.

**Discrepancy (a), pre-tokenised scripts: 483 of 1,095 (earlier LOG) vs 464 (feasibility note).**
- The earlier `s03_clean.py` summed split-contraction joins over the dialogue, action and *full* layers. Full
  already contains dialogue and action, so most joins were counted twice. That rule was re-run exactly
  (matches the saved per-script counts for 100% of scripts): 483 of 1,095, and 476 of the 1,087 after the
  exclusions (7 of the 8 excluded scripts are pre-tokenised).
- The same threshold on single layers gives 462 (dialogue only) and 471 (dialogue + action) of 1,087.
- The feasibility note's 464 can't be traced, because the note doesn't exist. **Our rule, fixed in the
  config:** ≥ 20 joins in the dialogue + action layer, which is 471 of 1,087. It is descriptive only, because the
  contraction fix is applied to every script anyway.

**Discrepancy (b), possible transcripts: 24 (earlier LOG) vs 55 (feasibility).** Both use the same rule
(action share < 0.15 or < 20 scene tags), on different denominators:
- 55 is over all scripts (55 of 1,095, and all 55 are among the 1,087).
- 24 is over the 526 scripts in the earlier definition-(C) comparison (bottom vs top quartile within cells).
- Checked against the saved `s10_flags.csv` and `script_table.csv`.

**Prior exposure (stated in the config).** `notebooks/scripts_and_reviews.ipynb` on `main` already compared a
speech-level VADER arc between hit and flop quartiles:
- "sentiment swing": p = 0.89
- "ending vs average": p = 0.41
- six k-means arc shapes: Kruskal-Wallis p = 0.99

Our primary descriptors are close relatives of the first two, so this pre-registration is not blind to that
earlier null. The time unit (scene, not speech bin), sample, controls and outcome (continuous) differ. The
franchise-free primary sample is likewise informed by the earlier branch's Step 10.

**Environment.** System Python 3.14 lacks VADER, spaCy, statsmodels, plotly and nbformat, and nothing may be
installed without permission. The course venv `02805-social-graphs/.venv-course` (Python 3.13.16) has every
package the earlier branch pinned, plus vaderSentiment 3.3.2, statsmodels 0.15.0, transformers 5.20.2 and
torch 2.14.1. It is used for everything here and pinned in the config. **No sentiment-tuned contextual model
is in the local Hugging Face cache.** Until the user allows a download, the contextual instrument is NOT
AVAILABLE, and under the fallback rule VADER is primary.

**Pre-registration.** `analysis/arcs/config.yaml` was committed before any sentiment score or association
was computed. The primary family is D1 (volatility) and D2 (ending minus start), with Bonferroni α = 0.025
each, on N = 834 (arc-ready, not likely franchise). r_min at 80% power is 0.097 at α = 0.05 and 0.107 at α = 0.025.

**Open questions for the user / group.**
1. May the contextual model (`distilbert-base-uncased-finetuned-sst-2-english`, about 270 MB) be downloaded?
   If not, VADER is primary and the contextual instrument is reported as NOT AVAILABLE.
2. Step 11 instrument match. The reviews side (`review_features.csv`) is the VADER compound of whole reviews,
   averaged over the first 50 per film. The script side would be VADER on sentences averaged per scene. Same
   tool, different unit. Does that count as "the same instrument and configuration"?
3. The work lives in a new worktree, `IMDb-movie-success-arcs` (branch `scripts-arcs` from `origin/main`).
   This session's own folder is a different repository.
