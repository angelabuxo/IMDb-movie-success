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

## 2026-11-14 · Step 1: sample and outcome

- All six funnel numbers reproduce from the raw data (2,446 / 1,095 / 1,087 / 868 / 946 / 759). Of the
  1,087 scripts, 219 are likely franchise (follows 146, named 133, both 60). 36 genre × decade cells; 23 primary
  films sit in cells with fewer than 5 films and are pooled.
- **Anomaly against the brief's expectation.** The hit lean of the script sample is real for all scripted films:
  - median multiple 3.69× against 3.41× for all clean films;
  - earlier quartile groups B 304 top vs 242 bottom, C 290 vs 236.
- **It mostly disappears once franchise films are out:**
  - primary sample B 235 vs 223, C 213 vs 220;
  - median 3.45×;
  - 7.3% of films gross less than their budget (all clean films: 6.9%);
  - SD of log(gross/budget) 1.04 (all clean films: 1.03).
- "Restricted range" is therefore mainly a property of the scripted sample as a whole, not of the primary
  sample. The notebook reading says so, rather than repeating the brief's claim. Selection is still not
  random (100k+ votes, public screenplay).
- SD of the outcome after cells and log budget: 0.91.
- Fix before commit: the outcome table's "median multiple" was first computed from the quartile midpoint. It is
  now the real median.

## 2026-11-14 · Step 2: text units

- 1,087 scripts, 0 parse errors, 148,637 scene tags.
- 456 scenes have no dialogue or description sentence and are dropped from the timeline (counted).
- 3,149,729 sentences: 1,675,413 dialogue and 1,474,316 action.
- Contraction fix: 155,109 joins; standalone "n't" tokens went from 73,083 to 1. 471 scripts are pre-tokenised (the rule fixed in Step 0).
- Arc-ready primary: 834, as expected. All 34 excluded films have fewer than 20 scene tags; none fails on
  sentences alone. Strict filter 812, loose 839; all scripts arc-ready 1,043.
- 9 possible transcripts remain in the primary sample (55 in all).
- Only 3 scripts number most of their headings. The feasibility note said 13, under a rule that can't be seen.
  So the order of appearance is the timeline for essentially every script: an unverifiable assumption.
- **Correction before commit.** The first version of the Step 2 text said a lexicon misses the negation in
  "do n't". Testing showed VADER scores split and joined forms identically on every hand-made case: its
  negation check matches any token containing "n't". Two consequences:
  - The text now says the fix is a safeguard for VADER, and that it matters for word counts and for a
    contextual tokenizer.
  - Step 3 will measure the effect on every affected sentence.
- Spotted in passing: "I didn't kill him" scores +0.58 in VADER. Negating a violent verb reads as positive. It
  goes into the Step 3 stress tests.
- Leak check: 0 of 20,000 random 8-word sentence openings appear in the committed outputs or notebooks.

## 2026-11-14 · Step 3 COMPUTE CHECK (stopped here for the user's approval)

| Item | Value |
|---|---:|
| Sentences, all 1,087 scripts ('both' layer; dialogue and action are subsets) | 3,149,729 |
| Sentences, primary 834 | 2,411,761 |
| Sentence length: median / 99th percentile / max words | 6 / 28 / 945 (15 over 350 words) |
| VADER: seconds per sentence (median of 3 timings on 200 seeded sentences; 20,000-sentence recheck: 8.1 µs) | 8.6 µs |
| VADER: projected full run, one core | ~0.01 h (under a minute) |
| Contextual (`distilbert-base-uncased-finetuned-sst-2-english`) | NOT AVAILABLE: not cached; no throughput measured, no runtime projected |
| Limit in config | 3.0 h |
| Memory | sentence table ~0.1 GB on disk; scores ~25 MB per instrument |

Fallback rule as of now: VADER primary. If the user allows the download, the model will be benchmarked on
the same 200 sentences. It becomes primary only if the projected full run is within 3.0 hours.

## 2026-11-14 · Config amendment: extra contextual configurations on DTU HPC (before any contextual score)

The user asked to use DTU HPC for heavier runs and allowed downloading the models there. Added to
`config.yaml` as `instruments.robustness_extra`:
- **`roberta3`** (`cardiffnlp/twitter-roberta-base-sentiment-latest`, three classes). Why: SST-2 has no neutral
  class and forces neutral action lines to a sign. About 60% of sentences are neutral to VADER (compute-check
  sample).
- **`scene_context`**: both models also score each scene's text in order, packed into pieces of at most 510
  word pieces, so the model sees context (week 7: chunking).

**Status.** These are robustness rows for Step 9 only. The pre-registered DistilBERT SST-2 remains the
contextual instrument of the fallback rule, and the primary family is unchanged.

**What had been computed before this amendment:**
- No contextual score.
- No association with any outcome.
- VADER only on the 200 + 20,000 sentences of the compute check (speed, share non-zero, mean), plus the
  hand-made contraction cases.

**Run notes.**
- The job measures its runtime projection on the HPC GPU and stops before the full run if it exceeds 3 h.
- The HPC venv has no pyarrow and nothing was installed, so the job reads gzipped JSON lines and writes
  gzipped CSV.
- Sentences are copied only to the owner-only folder `~/02805-embed/arcs/private` (mode 700, umask 077).

## 2026-11-14 · Contextual job submitted to DTU HPC

- Upload checked by SHA-256 (sentence table and job script identical on both ends). macOS tar metadata files
  were removed and the upload script now strips them.
- Both models downloaded to the scratch Hugging Face cache and smoke-tested on CPU with 200 sentences: SST-2
  gives {NEGATIVE, POSITIVE}, the RoBERTa model {negative, neutral, positive}.
- **Bug found before submission:** transformers 5 removed `build_inputs_with_special_tokens`. Pieces are now
  wrapped explicitly as [CLS] … [SEP] (BERT) or <s> … </s> (RoBERTa). On 50 sentences per model the
  wrapping matches the tokenizer's own output exactly. A CPU dry run of both code paths on 300 sentences, one
  of them longer than 510 pieces, ran with no NaN, with chunking and scene packing working.
- Job 30053002 is in queue gpua40 (one A40, walltime 6 h). It benchmarks first and skips the full run if the
  projection exceeds 3.0 h. No contextual score has been read back yet.

## 2026-11-14 · Config amendment (b): agreement rules for Step 3 (before any contextual score was read)

Added `instruments.agreement` while HPC job 30053002 was still queued, so nothing from it had been seen:
- labels for each instrument (SST-2 has no neutral class, so it is compared only on VADER-non-neutral sentences);
- the agreement statistics;
- the rule for the 20 private disagreement sentences.

Step 3 uses no outcome.

**Reading of the fallback rule's availability condition.** The config says contextual is primary only if it is
"available locally after the user answers the compute check". The model is not on this Mac. The user allowed it
on DTU HPC, and amendment (a) measures the runtime on the HPC GPU. We read a completed HPC run, within 3.0 h
projected, as meeting the availability condition. This is stated here rather than applied silently. The
decision is written to `outputs/s03_instrument_decision.json`, and Step 7 does not run without that file.

## 2026-11-14 · Step 3: instruments (VADER done; contextual pending)

- VADER 3.3.2 scored all 3,149,729 sentences (26 s on one core). Sentence scores are in `private/`; scene means
  per layer are committed as `outputs/s03_scene_vader.csv.gz`. The "both" scene score is the mean over all of
  the scene's sentences, not the mean of the two layer means.
- Primary 834: 61% of sentences are neutral (|score| < 0.05). Dialogue leans positive (24% vs 16%); action
  lines lean negative (20% vs 17%). Scene scores: median 0.00, 13% of scenes exactly 0. Per film, the SD of raw
  scene scores has a median of 0.124 (10th–90th percentile 0.088–0.168).
- **Correction to the Step 2 claim.** Hand-made cases had shown no VADER change from the contraction fix. On
  the real data, re-scoring all 690,303 sentences that contain a contraction in the split form changes 9,907
  scores (1.4%). The largest change is 1.51, and 4,395 scores cross the neutral band. In the 471 pre-tokenised
  scripts: 4,392 of 306,182.
  - Mechanism 1: the extra token shifts VADER's three-word negation and booster windows ("i'm", "it's", "don't").
  - Mechanism 2: a split possessive exposes a lexicon word ("god's" 679 of 679, "hell's", "friend's").
  - The fix keeps pre-tokenised and ordinary scripts on the same footing, which was its purpose. The Step 2
    notebook text already says Step 3 would measure this, and it is not changed.
- **Stress tests.** 17 constructed sentences; the column "constructed intent (author)" is not human
  validation. The contextual columns were scored on the HPC login node's CPU in fp32 with the job's own
  `Scorer` (`hpc/score_stress.py`).
  - SST-2 gives all three plain action lines a strong sign ("He opens the door and walks in." +1.00).
  - RoBERTa gives them P(neutral) 0.81–0.90.
  - "I didn't kill him.": VADER +0.58, SST-2 +0.99, RoBERTa −0.55.
  - VADER's "but" rule follows clause order (+0.57 vs −0.40).
  - All three instruments miss "love waiting in line all day".
- Agreement statistics and the 20 private disagreement sentences wait for job 30053002, which is still
  pending in gpua40. `ingest_contextual()` reads `private/contextual/meta.json`, merges one-to-one on
  (movie_name, scene, pos) with layer and row-count asserts, and writes `outputs/s03_instrument_decision.json`.
  Step 7 does not run before that file exists.
- Notebook 03 runs with 0 errors. Private cells have no output. Leak check: 0 of 19,995 eight-word openings
  found in the committed outputs, the notebook or the code.
- No outcome was used in this step.

## 2026-11-14 · Config amendment (c): how the validation sheets are drawn and scored (before any draw)

The config already fixed the sample sizes, labels and error categories. Added `validation.draw` and
`validation.scoring` before any sheet was drawn or any label existed:
- proportional genre allocation;
- blind labelling: the preview shows no scores;
- one neutral band for every instrument;
- the reference-label rule for two raters;
- accuracy with and without "mixed";
- kappa;
- the error table.

## 2026-11-14 · Step 4: human validation sheets (drawn; NOT YET VALIDATED)

- `validation/scenes_sheet.csv`: 50 primary-sample scenes, drawn by the rules in amendment (c). Drama 13,
  Comedy 9, Thriller 7, Adventure 6, Horror 5, Sci-Fi 5, Action 4, Other 1. Sizes run from 1 to 66 sentences
  (median 8). Columns: MovieSum name, scene index, genre bin, size, VADER scene score, empty contextual columns
  (filled from Step 3 when the HPC job returns), and empty rater columns.
- `validation/scripts_sheet.csv`: 25 random scripts from all 1,087, with the Step 2 flags and empty check
  columns (markup_ok, kind, id_mapping_ok).
- Private previews (gitignored) in `private/validation/`. The scene preview shows the text and no score, so
  labelling is blind. The script preview shows the first three scenes, the last scene, and the IMDb
  title/year/ID for the mapping check. **Interpretation:** the brief limits excerpts to ≤ 15 words. These
  previews hold whole scenes, because a rater can't label a scene from 15 words. They exist only on this
  machine for the raters and are never committed or published.
- Re-running the step never redraws the sample. It refreshes only the score columns and asserts the human
  columns are unchanged; a rerun is byte-identical.
- `python -m analysis.arcs.s04_validation score` applies the pre-registered scoring. It was tested on a
  synthetic filled copy in a scratch folder, never on the real sheet: unresolved pairs, adjudication, kappa,
  error table and the script checks all came out as expected.
- 0 human cells filled. Leak check on notebook 04, both sheets and the results: 0 of 19,995 sentence openings,
  0 IMDb-only titles, no IMDb IDs.
- Status: NOT YET VALIDATED. Every result built on the sentiment scores carries that caveat until the group
  fills the sheets.

## 2026-11-14 · Step 5: arcs (VADER; rebuilt with the contextual model once it returns)

- The arc maths is written as matrices (overlap-weighted resampling R, moving average S), so Steps 6 and 9
  can reuse it. Unit checks:
  - 3 scenes on 6 points map 2:2:2;
  - every row sums to 1 for 250 scenes;
  - a constant series gives D1 = 0 and D2 = 0;
  - a 0→1 step in the last 20% of scenes gives D2 = 1.
- All 1,087 scripts have an arc (834 primary). The median primary script has 133 scenes, about 1.3 per grid point.
- The average smoothed arc stays near zero: +0.015 over the first 20%, −0.004 over the last 20%, range
  −0.010 to +0.020. Medians: D1 0.038, D2 −0.018. 66% of films end lower than they start.
- **Proxy check, against my expectation.** I expected D1 to track scene length, since short scenes give noisy
  means. It does not: ρ = −0.01 with mean sentences per scene. Its strongest size correlate is total sentences
  (ρ = −0.22). All proxy correlations have |ρ| ≤ 0.22. The reading was first drafted with the expected
  direction and corrected to the data before commit.
- **Split-half reliability** (Spearman-Brown corrected): D1 0.53, D2 0.45, mean level 0.88. Roughly half of the
  between-film variance in the two primary descriptors is scene-level noise. That caps any association with
  profit (attenuation). Reported now, before any association is computed.
- The six example arcs are drawn with the config seed: 127 Hours, Barbarian, Hanna, House of 1000 Corpses,
  Next, The Accountant.
- No outcome used. Notebook 05: 0 errors; leak check 0 of 19,995.

## 2026-11-14 · Step 6: scene-order null (VADER)

- 1,000 shuffles of each script's scenes; one generator seeded with `seed`, scripts in sorted order. The
  rerun is identical, and the real D1/D2 match Step 5 to 1e-9 for every script. Runtime 2.5 s.
- **D1, one-sided:** 291 of 834 primary scripts (35%) beat their own shuffle null, against 5% by chance
  (binomial p 1e-158). Median z +1.14, 81% have z > 0, and the real D1 is typically 1.21× the shuffled mean.
  Scene order carries structure: neighbouring scenes share tone.
- **D2, two-sided** (distance from the null mean; this two-sided rule is fixed here and was not in the config):
  143 of 834 (17%) beat the null (binomial p 2e-37). Median z −0.52 in every genre bin. Endings are darker
  than a random placement of the same scenes, for more scripts than chance allows. For most single films the
  difference is within the shuffle range.
- All 1,087 scripts: the same picture (D1 35%, D2 18%).
- The 55 possible transcripts show much less order structure (D1 beyond null 11% vs 36%), which fits their
  having little description. Step 10 drops them as a check.
- The first draft of the reading called D2 "mostly within random ordering". It was rewritten from the data
  before commit: the excess over 5% and the negative lean are both clear.
- No outcome used. Notebook 06: 0 errors; leak check 0 of 19,995.

## 2026-11-14 · Second copy of the contextual job (queue backlog)

Job 30053002 waited more than 30 minutes for an A40 (232 jobs ahead), and every GPU queue was backed up. With
the user's approval, an identical copy was submitted to gpua10 as job 30053443 (`hpc/score_a10.lsf`). It
writes to `out_a10/`, so the two cannot overwrite each other.

**Rule:** whichever starts first is kept and the other is cancelled. Results are used from that job only, never
mixed. The runtime projection for the fallback rule is measured on whichever GPU ran (A40 or A10), and
`meta.json` records which.
