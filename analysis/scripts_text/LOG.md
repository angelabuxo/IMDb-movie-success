# Screenplay text analysis: decision and run log

Newest last. Each step records its numbers, anomalies, decisions and open questions.

## 2026-11-11 · Step 0: reproduce the coverage numbers

The coverage notebook (`01_coverage.ipynb`, `movie_script_table.csv`) named in the brief isn't on any
branch or on this machine, so the script table was rebuilt from the repo's own rules: the clean
theatrical rule in `scripts/imdb/coverage_clean_sample.sql`, MovieSum IDs with at least one `<scene>`
and one `<dialogue>` tag, and the longest script kept for the 12 duplicate IDs.

| Check | Expected | Reproduced |
|---|---:|---:|
| Clean 100k+ movies | 2,446 | 2,446 |
| … with a usable script | 1,095 | 1,095 |
| … grossing under budget | 67 | 67 |
| Bottom / top quartile of log(gross/budget) | 243 / 306 | 243 / 306 |

243 / 306 only reproduce when the quartile cut-offs are computed on all 2,446 clean movies and then
applied to the scripted ones (cut within the 1,095 they would be 274 / 274). That rule is now fixed in
`config.yaml` and used for definition (C) too. The imbalance is itself a finding: scripted films lean
towards hits.

Decisions fixed in `config.yaml` before any text was analysed:

- **Primary genre.** IMDb lists genres alphabetically, so "first-listed" means alphabetically first.
  We use the brief's seven genres in priority order, rarest first among the 1,095 scripts (Horror 137,
  Sci-Fi 140, Thriller 208, Adventure 286, Action 332, Comedy 338, Drama 564); 4 scripts have none of
  them and go to "Other". The reviews analysis uses first-listed; agreement is reported in Step 1.
- **Cells for (C) and the stratified null.** Primary genre × decade bin (pre-1980, 1980s … 2020s),
  quartiles cut on all clean movies in the cell, cells with fewer than 20 clean movies dropped and
  counted.
- **Score.** log10((rate_high + 0.1) / (rate_low + 0.1)), rates per 10,000 words; threshold log10(2).
  Same scale and tokenizer as `analysis/reviews` (on `reviews-exploration`); the 0.1 smoothing lets a
  word used by only one group have a finite score.
- **Primary analysis:** (C), dialogue layer, names removed, stopwords kept, no lemmatisation, minimum
  document frequency 5, repeated counts. Everything else is exploratory.

## 2026-11-11 · Steps 1-4

- **Step 1.** All five coverage checks reproduce. Budget bias is real: under (B) the median budget is $15M for
  the top quartile against $43M for the bottom (Mann-Whitney p = 5e-20), horror 17% vs 11%. (C) narrows it to
  $25M vs $40M (p = 0.0065), so budget is not fully controlled by (C); Step 4 matches on it. 32 of 47 genre ×
  decade cells have ≥ 20 clean films; 76 scripts sit in smaller cells and are out of (C). Single-stratum (C)
  is usable (≥ 30 per group) for Adventure, Comedy, Drama, Horror, 1990s, 2000s, 2010s only. The priority
  genre rule agrees with Àngela's first-listed rule for 49% of scripts: **open question for the group**,
  since the two halves of the paper should use one rule.
- **Step 2.** 0 parse errors, 0 empty dialogue layers. 44 scripts have < 20 scene tags (kept, flagged).
  **Anomaly → exclusion:** 8 scripts have < 500 tagged dialogue words because MovieSum's dialogue markup
  failed (Casablanca's whole dialogue layer is "THE END"; Air Force One has 2 words). They pass the coverage
  rule (one `<scene>` and one `<dialogue>` tag), which is therefore weak. Decision, taken before any word
  result: `text.min_dialogue_words: 500`, excluded from every comparison and counted. 3 of them were in (C).
- **Step 3.** 483 of 1,095 scripts are pre-tokenised (≥ 20 split contractions joined; 339,595 joins in
  all); before the fix "n't" was the 23rd most frequent word. Speaker cues include mis-tagged headings
  ("bedroom", "barn", "camera"), which is why name tokens that are dictionary words are removed only when
  capitalised mid-sentence. **Fix:** spaCy keeps periods in lemmas ("mr.", "l.a."), which inflated types
  in the lemma variant; lemmas are now normalised with the same tokenizer (no rerun of spaCy needed).
- **Step 4.** All 67 unprofitable scripts find a partner (caliper 0.5 SD of log budget). Max |SMD| on
  matched variables 0.28 → 0.03. IMDb votes are not matched on (post-release, partly the outcome); SMD stays
  −1.06: unprofitable films are far less watched.

## 2026-11-11 · Step 5 (primary run done; method unchanged after seeing it)

290 successful vs 236 unsuccessful scripts, 18,719 terms (df ≥ 5), 210 above 2×. The top terms are mostly
the failure modes the brief predicts: French dialogue in a few scripts (je, pas, c'est, vous), names that
escape per-script removal (rocky, balboa, jerry, eddie; mentioned in other films, dictionary words, or
sentence-initial), and franchise words (creed, stark, sparrow). Across the grid the top-50 lists barely
overlap the primary run (median Jaccard 0.02), the first sign of noise. **No method change was made in
response**: the primary analysis stays as pre-registered, and Steps 6, 8 and 10 test and expose these
effects. Explorers contain 0 of 117 probe runs of dialogue.

## 2026-11-11 · Steps 6, 7, 8, 10

- **Step 6 (primary test).** 210 terms above 2× against a stratified null of 182.0 ± 13.4 (1,000 shuffles,
  31 cells): z = 2.08, empirical p = 0.024. Unstratified: z = 2.04. The excess is on the successful side
  (z = 1.50, p = 0.073); the unsuccessful side is at chance (z = 0.04). No single term survives
  Benjamini-Hochberg or the family-wise (max-score) test. Read as **weak evidence**, pending Steps 8 and 10.
  Fix: the YAML key `null:` loaded as Python None; renamed `shuffle_null`.
- **Step 7.** 100 subsamples of the successful group down to 236: median top-50 Jaccard 0.69 (successful
  side) and 0.67; only 5 successful-side terms stay in the top 50 in ≥ 90% of runs. Matching is
  insensitive to order at calipers 0.5 and 1.0 (Jaccard 1.00); at 0.25, 61 pairs and Jaccard ~0.67.
- **Step 8.** 14 of the 30 most extreme terms are a speaker name in at least 3 scripts' cues (rocky, parker,
  jones, jerry, eddie, jimmy …). *Corrected: a first version counted a cue in any single script and reported
  23; MovieSum's cues include mis-tagged headings, so one script isn't enough.* per-script name removal misses names said in other films, names that are
  dictionary words, and sentence-initial names. The French words (je, pas, c'est, vous, il, ce, tu, dans)
  get 69-94% of their uses from one film, *Anatomy of a Fall*; "yah" 54% from *Fargo*. 20 of 30 terms get
  more than half their uses from one script. Human categories: not filled.
- **Step 10 (decisive).** 94 of the films in the comparison are likely franchise films (64 sequels by
  Wikidata "follows", 54 in a named franchise); 27% of the successful group against 7% of the unsuccessful.
  Without them the count test drops to z = 0.50 (p = 0.30); without sequels only, z = −0.12. The weak
  Step 6 signal is **not robust to removing franchise films**. Without the 24 possible transcripts: z = 1.51.
  *Corrected: a first version said the signal "reflects sequels' shared names"; that mechanism isn't tested,
  so the wording now states only what was measured.*

## 2026-11-11 · Steps 9, 11, 12

- **Step 9.** 526 scripts, 10 × 5-fold CV: TF-IDF text AUC 0.584, baseline (genre + decade + log budget)
  0.576, both 0.574. The text model beats its own shuffled-label null (0.491 ± 0.032, 200 shuffles,
  p = 0.010), so the dialogue carries some signal, but its top coefficients are the same names, franchise
  and French words, and text vs baseline is **not clearly different** (repeat means overlap: 0.557-0.612 vs
  0.559-0.587; text wins 52% of folds). Wording fixed so the reading doesn't say "beats" on a 0.008 gap.
- **Step 10 addition.** Without franchise films the classifier gives text 0.576 vs baseline 0.662 (3
  repeats): the text falls clearly below genre, decade and budget.
- **Step 11.** Sheets for 25 scripts and 30 terms written with empty columns; status NOT YET VALIDATED.
- **Step 12 finding candidate (a negative result with its checks):** within genre and decade, the most and
  least profitable films' dialogue differs only slightly more than random groupings (z = 2.08, p = 0.024),
  the excess disappears without franchise films (z = 0.50, p = 0.30), and a TF-IDF classifier doesn't beat
  genre, decade and budget (0.58 vs 0.58; without franchise films 0.58 vs 0.66).

## 2026-11-11 · Review fixes (before any push)

- **Dates.** The log and config first said 2026-11-12; the work was done on 2026-11-11. Corrected. The seed
  value `20261112` is left unchanged, because changing it changes every result.
- **Matching after exclusion.** Matching now runs on the 1,087 scripts left after the 8 failed-markup
  exclusions, so the balance table describes the sample that is analysed. Still 67 of 67 matched; max |SMD|
  on matched variables 0.27 → 0.03.
- **No IMDb fields in committed tables.** `script_table.csv`, the Step 10 flags and the validation sheet now
  identify films by MovieSum's `movie_name` and carry genre and era only as derived bins; IMDb titles, raw
  genres, exact years and votes stay in `private/` (IMDb's terms don't allow republishing them).
- **Franchise control (exploratory).** Dropping the same number of random non-franchise films (77
  successful, 16 unsuccessful), 50 times, each against its own 200-shuffle null: median z 2.94 (90% between
  1.75 and 4.36), and 0 of 50 fall as low as dropping the franchise films (z = 0.50). So the excess goes with
  those particular films, not with losing sample size.
- **Residue in the term lists.** `ll`, `ie`, `eo` and `wi` among the top terms look like tokenisation or
  encoding residue (split "'ll", initials, broken accents). Noted; the primary analysis is unchanged.
