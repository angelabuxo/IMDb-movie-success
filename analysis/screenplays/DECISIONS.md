# Screenplay analysis (Q3): decisions and evidence

**Question.** Group project (DTU 02805). Main question: what best predicts whether a movie earns a lot compared with its budget: its genre, the people who make it, or its script? This folder answers the script part (Q3): within one genre, do hit scripts differ from flop scripts?

**Who decides.** Every decision below was taken by the group. Each has a reason, and where a number supports it the evidence is in the appendix. Nothing here was left to a tool to decide.

**Hand-in.** Friday 27 November 2026.

**How to read this file.** Section 1 is the summary. Section 2 lists every decision (D1 to D28) with its reason. Section 3 lists the limitations. The appendix holds the evidence tables. Rules that were fixed before seeing a result are marked **(before results)**. Choices made after looking at results are marked **(post hoc)** and are exploratory.

## Contents

1. [At a glance](#1-at-a-glance)
2. [Decisions](#2-decisions)
3. [Limitations](#3-limitations)
4. [Status and next steps](#4-status-and-next-steps)
5. [Background: what Word2Vec gives us](#5-background-what-word2vec-gives-us)
6. [Appendix: evidence](#6-appendix-evidence)
7. [Rules for this folder](#7-rules-for-this-folder)

## 1. At a glance

### From scripts to the four tests

| Step | Films |
|---|---:|
| Unique MovieSum scripts (one per IMDb id) | 2188 |
| ... of them in the 10k+ vote table | 2020 |
| ... that pass the shared money rule | 1621 |
| ... and have at least 500 tokens in both layers (**final set**) | **1598** |
| ... of which in the 100k+ vote tier (robustness check) | 1042 |

Hit and flop groups (equal sizes by construction, middle half dropped): **Comedy 136 / 136**, **Action 108 / 108**, Drama 223 / 223 (exploratory).

### The design in one paragraph

Each script becomes two vectors, one for the dialogue and one for the scene description: the TF-IDF-weighted mean of Word2Vec word vectors, from a model trained on all 2188 scripts. A logistic regression predicts hit versus flop inside one genre and one era. We report the AUC in repeated cross-validation, and compare it with a shuffle of the labels, a TF-IDF representation, and a metadata baseline (vote tier, era, other genres). The hypothesis, written before any result: dialogue separates Comedy hits from flops better than description does, and the reverse for Action.

### Decision index

| ID | Topic | Decision in one line |
|---|---|---|
| D1 | Approach | Script vector from Word2Vec, always compared with simpler baselines |
| D2 | Layers | Dialogue and description are tested separately; both together is only a comparison |
| D3 | Discipline | One primary analysis and the hypothesis fixed before results; the rest exploratory |
| D4 | Checks | Label shuffle, TF-IDF, metadata baseline, hand validation of the layer split |
| D5 | Universe | All films of the 10k+ vote table with a MovieSum script (1598); 100k+ only as a check |
| D6 | Money rule | One shared rule, `usable_money`, chosen on a principle and not on outcomes |
| D7 | Shared files | One analysis table for the three questions (`analysis-set` bundle) |
| D8 | Duplicates | Keep the longest script of a duplicated IMDb id |
| D9 | Tiny layers | Exclude films with fewer than 500 tokens in either layer |
| D10 | Vote tier | No control in the label; state the limitation; tier-only baseline; 100k+ check |
| D11 | Hit and flop | Top and bottom quarter of ln(gross/budget) within genre and era; middle dropped |
| D12 | Era | Cut within genre x era; six era bins; cells under 8 films excluded |
| D13 | Multi-genre | A film counts in every genre it has |
| D14 | Group size | At least 50 hits and 50 flops for a main result |
| D15 | Genres | Comedy and Action primary, Drama exploratory |
| D16 | Layers from tags | `dialogue`, `scene_description` kept; parentheticals out |
| D17 | Parser | No cleaning in the parser; headings inside dialogue stay |
| D18 | Symbols | Non-letter symbols removed in preprocessing; no exclusion for garbled text |
| D19 | Training corpus | Own corpus, all 2188 scripts, one space for both layers |
| D20 | Word2Vec settings | Skip-gram, 100 dimensions, window 5, `min_count` 20, 5 seeds |
| D21 | Aggregation | TF-IDF-weighted mean (primary), plain mean (robustness) |
| D22 | Preprocessing | One tokeniser for both layers; each element is a context unit |
| D23 | Character names | Removed only if the token is specific to fewer than 20 films |
| D24 | Test settings | Classifier, cross-validation, baseline, shuffle and bootstrap fixed before any AUC |
| D25 | Hypothesis | Comedy: dialogue better; Action: description better |
| D26 | Franchises | Robustness check only, with the raw flag |
| D27 | Manual check | 25 films, stratified by era, error rates per era and layer, never pooled |
| D28 | Follow-ups | 100k+ shuffle, vote check, word lists: post hoc and exploratory |
| D29 | Budget-matched similarity test | Post hoc: cosine and label shuffle within genre x era x budget, criteria fixed before the five-seed run |

## 2. Decisions

### A. Design

#### D1. Approach
- **Decision.** Represent each script as a Word2Vec-based vector and ask how much it tells about success. Always compare with baselines.
- **Why.** A vector is only informative relative to simpler alternatives.

#### D2. Layers
- **Decision.** Analyse dialogue and scene description separately (the 4 tests). The combined text is a third, comparison analysis only.
- **Why.** They are different registers. Mixing them hides what drives a result and brings in the dialogue/description proportion, which is close to a genre variable.

#### D3. Discipline
- **Decision (before results).** One primary analysis, fixed before seeing results; everything else is labelled exploratory. The hypothesis is written before any result.
- **Why.** Many variants means some look significant by luck, and a story fitted to the outcome is not evidence.

#### D4. Checks
- **Decision.** Shuffle the hit/flop labels (null); TF-IDF as a baseline representation; a metadata baseline; hand validation of the layer split; a nearest-neighbour sanity check of a few words (planned; not reported in this folder).
- **Why.** Each one tests a different way the result could be fake or trivial.

### B. Data

#### D5. Universe
- **Decision.** All films of `movies_features_10k.csv` that have a MovieSum script, pass the money rule (D6) and the layer rule (D9): **1598 films**. The 100k+ vote tier alone (1042 films) is a robustness check.
- **Why.** (1) Group sizes are comfortable (Thriller is exactly 50 at 100k+, a knife edge). (2) Selection on IMDb votes is partly selection on success: a flop inside the 100k+ tier is a film with many votes but a low return. The 10k tier reduces this bias **without removing it**. (3) It is the universe the other two questions already use, so the common set is much larger.
- **Evidence.** Appendix A4.

#### D6. Money rule
- **Decision.** One shared cleaning rule for money: `analysis.clean_rule` (called, never rewritten), **and** no `net_gross` flag, **and** no flag on the budget. A film that passes is `usable_money`. The rule requires budget and gross in US dollars, at least $1,000 each, drops grosses flagged as rentals, regional or direct-to-video, drops films of a streaming distributor, and keeps only films released 1975 to 2025.
- **Why.** The principle: a gross that is not comparable with a worldwide theatrical gross makes a film look like a flop through measurement, not performance. That covers flagged infobox grosses, films before 1975 (the README says their grosses are mostly distributor rentals; **this claim has not been checked**) and streaming-distributor films (28 films in the 100k+ scripted set, all from 2016 on). **The rule was chosen on this principle and not by looking at hits or flops.**
- **Replaces.** An earlier rule that excluded any infobox flag (25 films).
- **Evidence.** Appendix A4.

#### D7. Shared files
- **Decision.** The decisions are built into one shared table by `scripts/imdb/build_analysis_set.py` and shared as a dated bundle (`analysis-set-2026-11-19.tar.zst`; data dictionary in `docs/analysis-set.md`). `film_analysis_set.csv` has every film of the 10k+ table with its flags and exclusion reasons; `in_common_set` is the set on which the three questions are compared head to head; `usable_ratio_same_currency` allows questions that only need the ratio to keep 607 more films. No hit/flop labels, token counts or script text are in it.
- **Why.** One filter for all three questions keeps the sets from being changed silently, and lets the earlier analyses be re-run on the same table.

#### D8. Duplicated IDs
- **Decision.** For the 12 duplicated IMDb ids, keep the record with the longest script (most characters). The `* 2.jsonl` copies are ignored (identical to the originals). Implemented in `parse_layers.one_per_id`.
- **Why.** The ids are different versions of the same film, so one has to be chosen; longest means most complete.

#### D9. Films with a tiny layer
- **Decision.** Exclude, from the whole analysis (both layers), the films with fewer than 500 whitespace tokens in the dialogue layer or in the description layer (rule in `data_check/layer_stats.py`). The exclusion is applied before the hit/flop quartiles are cut, so hits and flops stay balanced in each cell. The value 500 was a first choice, not a tuned one.
- **Why.** (1) A script vector is the mean of the word vectors of a layer; with no tokens there is no mean. (2) The dialogue-versus-description comparison only makes sense on the same films. (3) Among the 18 films with almost no tagged dialogue, the description layer is about twice as long as usual, which suggests dialogue sitting untagged inside it; this was a hypothesis from counts, and the exclusion does not depend on it (either a layer is missing or its content is doubtful). (4) The 12 films with an empty or tiny description have many `character` elements but 0 or 1 `scene_description`. (5) The rule uses only the structure of the scripts, never the outcome.
- **Evidence.** Appendix A2, A3 and A6 (manual check of the tiny-layer films).

#### D10. Vote tier
- **Decision.** No control for the vote tier in the definition of hit/flop. Instead: (i) state the limitation; (ii) include a tier-only baseline (tier, genre and era alone) in the comparisons; (iii) repeat the main analysis on the 100k+ films only.
- **Why.** Hits are 82 to 96 % 100k+ films and flops 26 to 57 %, and votes are partly a consequence of success, so conditioning on the tier would condition on a consequence of the outcome. Cutting quartiles within genre x era x tier leaves too few films per group.
- **Evidence.** Appendix A5.

### C. Labels

#### D11. Hit and flop
- **Decision.** Relative within the genre: top quarter versus bottom quarter of ln(gross/budget). The middle half is dropped.
- **Why.** The absolute rule "gross at least 2x budget" gives almost no flops in this sample (Action: 248 hits, 18 flops) and answers a different question. Dropping the middle leaves half of each genre's films.
- **Note.** Gross over budget is not profit.

#### D12. Era
- **Decision (rule written before seeing the decade table).** If the share of hits under the within-genre cut is roughly flat across decades, keep that cut; if not, cut quartiles within genre x decade. The table showed it is far from flat, so **quartiles are cut within genre x era cells**. Era bins: before 1980, 1980s, 1990s, 2000s, 2010s, 2020s; 1975-79 is **not** merged into the 1980s. **Minimum cell size 8 films**; smaller cells are excluded from the primary analysis.
- **Why.** The hit share depends strongly on the decade (about 70 % in the 1970s, 12 to 14 % in the 2000s and 2020s, when 25 % is expected), in all six large genres. Cutting inside era removes era from the contrast by construction, so era-specific vocabulary and script formats cannot be read as success differences. Merging 1975-79 into the 1980s would bring era back inside a bin. A quartile on 4 to 7 films gives groups of 1 or 2 films.
- **Evidence.** Appendix A1.

#### D13. Multi-genre films
- **Decision.** A film counts in every genre it has. "Genre X" means "film tagged X". Whenever more than one genre is analysed, the overlap is reported (for example Comedy and Action share 55 films; `genre_counts/genre_overlap.csv`, to be recomputed on the final set).
- **Rejected.** Only the first-listed genre: the `genres` column is in alphabetical order, so "first" means first in the alphabet, not "main" (Thriller gets 0 films, Drama 48). Only single-genre films: only 74 films qualify and Action has none.

#### D14. Group size
- **Decision.** At least 50 films in each of the hit and flop groups for a genre to be a main result.
- **Why.** With small groups a null result is not informative: there may be a difference we cannot see. The label shuffle protects against false positives, not against low power.

#### D15. Genres
- **Decision.** Primary: **Comedy and Action**, each in the dialogue and in the description layer (4 tests; all four are reported, never only the best). **Drama** is an exploratory check. No other genre is analysed.
- **Why.** A hypothesis-driven choice with enough films per group (Comedy 136, Action 108; Drama 223 is the largest genre, so a check with maximum power). Results for different genres are not independent (D13).

### D. Parsing and cleaning

#### D16. Layers from tags
- **Decision.** `dialogue` is the dialogue layer, `scene_description` the description layer, `parenthetical` its own layer (excluded from both main layers), and `character` and `stage_direction` are "other".
- **Why.** Parentheticals such as "(exasperated)" are not spoken, so adding them to the dialogue would contaminate "what the audience hears".

#### D17. No cleaning in the parser
- **Decision.** Every element stays in the layer its tag says. Junk (non-header `stage_direction`, numeric or empty `character`, empty tags) is counted as possible leakage in the statistics. **Headings (INT./EXT.) inside dialogue are not cleaned and no sensitivity analysis is run.**
- **Why.** `character` and `stage_direction` are outside both analysis layers, so that noise does not enter the vectors; a filter rule would be one more thing to defend. Headings are 0.10 % of the dialogue elements (1418 of 1,440,867), in 177 of 1598 films (37 films above 1 %).
- **Evidence.** Appendix A6 and A7.

#### D18. Non-letter symbols
- **Decision.** Symbols that are not letters are removed in the preprocessing. Corrupted words are not repaired, and no film is excluded for garbled text.
- **Why.** This removes scan symbols (bullets, tildes, backslashes) and markup residue. The share of symbols has no gap (150 films above 1 %, a continuum) and does not isolate the two films called garbled in the manual check (ranks 1 and 113 of 1598), so it cannot support an exclusion rule. Rare tokens from corrupted words are dropped by `min_count` (D20).
- **Evidence.** Appendix A7.

### E. Representation

#### D19. Training corpus
- **Decision.** Word2Vec is trained on the 2188 unique scripts, in one space for both layers, on our own corpus. GloVe is an optional robustness check.
- **Why.** No label is used, so no outcome leaks into the vectors (the test films do take part in the training: a limitation). A pretrained space describes its own corpus (Wikipedia, news), not screenplays. Size: 48.3 million whitespace tokens (20.2 M dialogue, 28.2 M description; part of the description count is punctuation split off).

#### D20. Word2Vec settings
- **Decision.** Skip-gram with negative sampling, 100 dimensions, window 5, `min_count` 20, 5 seeds with `workers=1`. Robustness: windows 2 and 10.
- **Why.** A narrow window gives more syntactic information, a wide one more topical. The corpus is small, so no more dimensions. `workers=1` makes a run reproducible. Seeds are reported as mean and spread of the AUC; **vectors of different seeds are never mixed** (each space has its own axes). `min_count` 20 drops rare words, including corrupted ones, but also some legitimate rare vocabulary.

#### D21. Aggregation
- **Decision.** TF-IDF-weighted mean as the primary vector, plain mean as robustness. The IDF is computed over the 2188 scripts, **per layer** (a document is the layer of one film).
- **Why.** A plain mean is dominated by frequent words (section 5). IDF = ln(N/df) gives weight 0 to words in every script. Per layer, so the documents are comparable within a layer.

#### D22. Preprocessing
- **Decision.** Lowercase; non-letter symbols removed (D18); no lemmatisation; stopwords kept; one tokeniser for both layers that joins detached contractions ("do n't" to "don't") and removes attached punctuation; each element (one line of dialogue or one description) is its own context unit, so the window never crosses elements; parentheticals out of both layers.
- **Why.** Word2Vec needs the context, and the IDF weighting already reduces stopwords. A different tokenisation per layer could turn a "difference between layers" into an artefact.
- **Not discussed explicitly, to be reviewed.** Digits are removed too (they are not letters), hyphenated words are split, and an apostrophe inside a word is kept.

#### D23. Character names
- **Decision.** The film's own character names (from its `<character>` tags) are removed from both layers, **only when the token is a character-label token in fewer than 20 of the 2188 films** (`NAME_MAX_FILMS` = 20, about 1 % of the films).
- **Why.** Removing every label token took 10.8 % of the dialogue tokens and 17.3 % of the description tokens, because labels are often ordinary words (`voice` is a label in 1419 films, `man` in 1264, `the` in 1112; `guard`, `officer` and `doctor` in 400 to 580). That deletes the word wherever the label is used and unequally across layers, which is the artefact the single tokeniser was meant to avoid. With the filter it is 1.47 % and 4.2 % (1305 of 34,020 label token types are kept as common words); names shared by many films are reduced by the IDF.
- **The value 20 is arbitrary; there is no natural cut** (names and ordinary words mix: `john` is a label in 233 films, `police` and `door` in 100 to 300). It was fixed before any AUC was computed, the only criterion is "the token is specific to a few films", and it is not to be changed after seeing results.
- **Numbers.** `preprocess/preprocess_summary.csv`.

### F. The test

#### D24. Test settings fixed before any AUC
These settings were written and committed before any AUC was computed; `word2vec/` implements them and they are not to be changed after seeing results.

| Setting | Value |
|---|---|
| Word2Vec details not fixed above | gensim 4.4.0 defaults for everything not listed: 5 epochs, `negative` 5, `sample` 1e-3, `alpha` 0.025. One corpus: every kept element of both layers of the 2188 scripts. Primary seeds 1-5 at window 5; robustness windows 2 and 10 with the same seeds. |
| Labels | `final_set.final_films()` and `genre_cells.label(pairs_for(films, "a"))`; hit vs flop only, middle dropped. Comedy 136/136, Action 108/108, Drama 223/223 (exploratory). |
| Script vector | Per film and layer, the mean of the word vectors weighted by tf x idf, tf = count of the word in that layer of that film, idf = ln(N/df) over the 2188 layers of that layer. Words outside the `min_count` vocabulary are dropped. A film whose weights sum to 0 stops the run. |
| Classifier | L2 logistic regression on standardised features; C picked inside each training fold by 5-fold inner CV on AUC over `logspace(-4, 2, 7)`. |
| Cross-validation | Repeated stratified 5-fold, 5 repeats, `random_state` 2805. The same folds for every representation and both layers within a genre, so comparisons are paired. AUC = AUC of the pooled out-of-fold scores of one repeat, averaged over repeats; for Word2Vec also averaged over the 5 seeds (one classifier per seed, vectors never mixed). |
| Baseline "tier + genre + era" | Features: tier (100k+ = 1), era one-hot, multi-hot of the film's other genres. Same classifier and folds. Era is balanced by construction (quartiles within genre x era), so tier and co-genres carry the baseline. Also reported: vector + baseline features together. |
| TF-IDF representation | The same tokens, restricted to the Word2Vec vocabulary, weights tf x ln(N/df) as above, rows L2-normalised; same classifier (no standardisation) and folds. Only the representation differs from the Word2Vec vector. |
| Label shuffle | 200 permutations of the hit/flop labels within the genre; each permutation reruns the identical pipeline (all 5 seeds averaged for Word2Vec). p = (1 + #null >= observed) / 201. |
| Hypothesis contrast | AUC(dialogue) - AUC(description) per genre, with a 95 % interval from 2000 paired bootstrap resamples of the films (the same films for both layers), on the out-of-fold scores, averaged over seeds and repeats. The interval ignores training variance (stated). Prediction: Comedy > 0, Action < 0. |
| Robustness | Windows 2 and 10; plain mean instead of TF-IDF weighting; 100k+ tier only. Reported after the primary results, not used to choose them. |

#### D25. Hypothesis
- **Decision (before results).** Comedy: the difference between hits and flops is larger in the dialogue layer than in the description layer. Action: it is larger in the description layer. Measure: the AUC of hit versus flop from the script vector, per genre and layer; the quantity compared is AUC(dialogue) minus AUC(description). Each AUC is compared with (a) the baseline "tier + genre + era", (b) TF-IDF and (c) the label shuffle. All 4 tests are reported, nulls included.
- **Why.** The conjecture (not a result): humour lives in how people speak, action spectacle in what is described. A vector that does no better than the tier baseline says nothing about the script.

#### D26. Franchises
- **Decision.** Only a robustness check: repeat the analysis without the films flagged as franchise, with the raw flag.
- **Why.** The Wikidata series property is not a clean franchise flag, so this is stated as a limitation.

### G. Validation

#### D27. Manual check
- **Decision.** 25 films drawn at random (seed 2805) from the final set, stratified by era: 10 from before 1990 and 15 from 1990 on. Per film about 10 points: 4 dialogue, 4 description, 2 from other. Each point is one element of at most 15 words, in the layer the parser gave it, with the tags (not the text) of the previous and next element. Error rates are reported per era group and per layer and are **not** pooled. Two separate sheets, never mixed into the rates: a targeted red-flag sheet (exploratory, about 5 suspicious elements per film) and a sample of the tiny-layer films (5 dialogue-poor and 3 description-poor films, 10 elements each).
- **Why.** The excluded films are concentrated before 1990 (37 % against 16 %), so tagging quality may depend on era; a plain random sample would hold only about 3 old films. Dialogue and character elements outnumber descriptions about two to one, so a plain sample gives too few description points. The tiny-layer sample tests directly whether dialogue sits untagged inside the description.
- **Evidence.** Appendix A6.

### H. After the primary results

#### D28. Follow-ups **(post hoc)**
Requested after the four primary tests had been seen, so they are exploratory:
- **Shuffle inside 100k+** (`word2vec/subset_tests.py`): the primary pipeline unchanged, 200 permutations, for the four tests, Word2Vec and TF-IDF. The baseline loses its tier column there (constant), so it is genre + era. The folds come from the smaller set, so they differ from those of the primary table.
- **Interval for vector + baseline** (same script): paired film bootstrap, 2000 resamples of the out-of-fold scores, for the AUC and the gain over the baseline. No refit inside the bootstrap.
- **log(votes) check** (`word2vec/votes_check.py`): votes still vary more than tenfold inside 100k+, so the tier split leaves popularity in. Observed AUCs only, no shuffle. Votes are partly the outcome (more viewers, more gross), so this measures how much of the separation popularity can carry; it is not a confounder to adjust away.
- **Word lists** (`words/words.py`): the log-odds ratio with an informative Dirichlet prior (the statistic behind Scattertext's `LogOddsRatioInformativeDirichletPrior`, hand-written because Scattertext does not install on Python 3.13), prior = counts over the 1598 films, minimum 10 labelled films per word, 1000 label permutations within genre x era cells. Fixed before looking at any list. **Changed after looking:** the lists are ranked by the shuffle-calibrated z, because the plain z ranking was led by single letters from a few garbled scripts and by names the 20-film filter keeps. The plain ranking is still in the CSVs.

#### D29. Budget-matched similarity test **(post hoc)**
Chosen on 2026-11-25, after the planned test (D24, D25) and after a one-seed run of this test had been seen. It is exploratory, and D24 stays in the paper. This section was written before the five-seed results and the Horror checks were run.

- **Question.** Do hit scripts resemble each other, and flop scripts each other, more than hits resemble flops, among films of the same genre, era and budget group?
- **Why change the test.** The paper asks what facts known before release go with return. The D24 baseline uses the vote tier, which is known only after release. The genre and people questions already hold budget fixed. Cosine similarity and a label shuffle are course methods.
- **Main test.** Vectors, labels and statistic as in `all_genres.py`, window 5, seeds 1 to 5. Labels are shuffled 2000 times within genre x era x budget tercile (terciles of all films of the genre x era cell). Nine genres (D14) x two layers = 18 tests, Holm-corrected. A test passes at Holm p < 0.05. All 18 are reported.
- **Check of the main test.** The same test with budget halves and quarters, and a check that the era-only run reproduces `all_genres.csv`.
- **Checks for any genre that passes.** Holm is not applied to them (they are checks on a passed test, p < 0.05): (1) without the Wikidata franchise flag, (2) 100k+ films only, (3) shuffled within era x budget tercile x vote tier, and (4) each genre's separation is read against the share of films in strata the shuffle can move.
- **How it will be read, fixed in advance.** A genre is reported as a finding only if it passes the main test and the halves and quarters. If it fails the franchise check in a layer, that layer is reported as tied to franchises. If the 100k+ check has fewer than 20 hits or flops, it is reported as too small to read. Everything that fails is "no signal found with these vectors", never "scripts do not matter".
- **Words.** For a passing genre, log-odds words (`words.py` statistic) with labels shuffled within era x budget tercile. If no word beats the family-wise shuffle threshold, the paper says the test cannot say what differs.
- **Votes.** Only in check (2), (3) and in the coverage tables, as a check.

## 3. Limitations

To be stated in the paper.

- **One annotator** for the manual check, 25 films, snippets of at most 15 words, points clustered by film (the intervals are optimistic).
- **Gross over budget is not profit**: marketing is missing and cinemas keep part of the gross.
- **Votes are partly a consequence of success.** The 10k tier reduces the selection bias without removing it; inside 100k+, hits still have far more votes than flops, so the text signal cannot be separated from "a widely seen film".
- **Era is controlled in decade bins only**, so a recent word can still carry the newest films inside a bin.
- **The test films take part in the unsupervised training** (no labels, so no outcome leaks).
- **A character name that is also an ordinary word** and a label in fewer than 20 films (Will, Grace) is removed from that film's text; common first names shared by 20 or more films stay in; the cut of 20 is arbitrary; labels such as `int`, `ext` and `cut` show that some headings are tagged as characters.
- **`min_count` 20** also drops some legitimate rare vocabulary.
- **Headings (INT./EXT.) inside dialogue** are not cleaned: 11 % of films have at least one.
- **Franchises** are flagged with the raw Wikidata flag, which is noisy.
- **The bootstrap intervals ignore training variance**, so they are too narrow if anything.
- **The follow-ups are post hoc** (D28).
- **A weak, spread-out signal.** With 108 to 136 films per group the tests cannot see small differences, and a null result is not proof of no difference.

## 4. Status and next steps

- **Done:** data checks, rules, the manual check, preprocessing, the four tests with their checks, the follow-ups, candidate figures (`figures/`, see its README for which one to use).
- **Pending:** an independent reproduction of the AUCs, and a check that the "Exception ignored" lines in the Word2Vec training logs are harmless. The word-list results were reproduced exactly.
- **Pending:** decide with the group how the Q3 result fits the comparison of the three questions on the common set.
- **Then:** the paper text (what each measurement captures and what it misses, with the checks next to each claim).

## 5. Background: what Word2Vec gives us

- **One vector per word, not per script.** Word2Vec slides over every token position in the training text and learns to predict neighbouring words (skip-gram, negative sampling). The vectors are a by-product. Words in similar contexts get similar directions.
- **Unsupervised.** It knows nothing about hit/flop or money. Success enters only when we compare script vectors with labels.
- **Static.** One vector per word type, so no polysemy. The axes have no individual meaning.
- **A script vector is built by us**, as a weighted mean of the vectors of all word occurrences in the script. It captures what kind of contexts the vocabulary comes from.
- **What the script vector loses:** word order, plot, arc, rare words (diluted by the mean), polysemy. It does not reliably capture sentiment (hero and villain have a high cosine).
- **Cosine of two vectors** means similar contexts. It does not mean shared vocabulary or the same meaning.
- **A plain mean is dominated by frequent words.** Toy example: adding "the" three times to an action and a romance script moved their cosine from 0.33 to 0.75. So aggregation is a decision (D21).

### Questions the vectors cannot answer

- Which words drive a difference. That needs a word-level method (the word lists in D28).
- Profit. The vector alone says nothing about it.

## 6. Appendix: evidence

Tables, with the script that produced each. Only counts, tags and derived numbers are shown; no script text.

### A1. Share of hits per decade (quartiles cut within genre only)

About 25 % everywhere if the decade did not matter. From `genre_counts/decade_hits.csv` (script `genre_counts.py`); n in brackets (film-genre pairs).

| | 1950s | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|---|---|
| All genres pooled | 58.3 (12) | 69.7 (33) | 70.8 (65) | 50.3 (177) | 30.6 (527) | 13.5 (779) | 24.9 (1105) | 12.5 (208) |
| Drama | 50.0 (4) | 57.1 (7) | 66.7 (18) | 39.1 (23) | 35.0 (103) | 14.5 (145) | 24.1 (232) | 12.8 (47) |
| Comedy | | 50.0 (2) | 100 (5) | 41.9 (31) | 30.8 (65) | 15.0 (80) | 24.6 (130) | 8.0 (25) |
| Action | | 100 (3) | 100 (2) | 80.0 (20) | 25.8 (62) | 13.3 (90) | 21.3 (127) | 27.3 (22) |
| Adventure | | 100 (4) | 40.0 (5) | 58.3 (24) | 34.7 (49) | 16.4 (67) | 17.5 (114) | 13.0 (23) |
| Crime | 100 (2) | 66.7 (3) | 85.7 (7) | 46.7 (15) | 30.4 (46) | 15.5 (71) | 20.3 (64) | 6.7 (15) |
| Thriller | 66.7 (3) | 100 (3) | 60.0 (5) | 60.0 (10) | 17.9 (39) | 10.4 (67) | 34.3 (70) | 8.3 (12) |

(Computed on the superseded 1104-film set; not recomputed under the shared money rule. 1930s and 1940s have 5 and 4 pairs and are in the CSV.)


### A2. The 30 films with an empty or tiny layer

From `data_check/tiny_layers.csv` (tags and counts only, no text). A layer is "tiny" below 500 tokens. 21 of the 30 were in the superseded 1104-film set; 19 of them in the 100k+ set under the shared rule (see below).

- **Description tiny or empty (12 films, e.g. Gandhi, Wild Wild West, The Next Three Days).** The film has hundreds of `character` elements, so the dialogue is there, but there are zero or one `scene_description` elements, and often many empty `<scene />` markers (up to 186). The scene content was probably lost or never tagged.
- **Dialogue tiny or empty (18 films).** Few or no `character` elements (0 to 9, one case 43) while the description layer is large: median 26,700 tokens against a median of about 12,400 for all films. This suggests that in these films the dialogue sits in the description layer untagged. If so, the description layer is contaminated in the same films, which the name and heading proxies would not catch. Two of them (All Is Lost, The Red Turtle) are known to have almost no dialogue and are probably genuine.
- This is a hypothesis from counts. The manual check is what can confirm it.


### A3. Cost of the exclusion and who is excluded (superseded 1125 / 1104-film set)

From `genre_counts/tiny_vs_rest.py` (`tiny_vs_rest.csv`). Only year, genre and the tiny flag are used, never outcomes.

- **Size:** 21 of 1125 films (1.9 %).
- **Per genre (films tagged):** Drama 11, Comedy 9, Action 7, Adventure 3, Crime 3, Thriller 1. After the exclusion the six main genres still have 147, 87, 82, 74, 58 and 54 hits and flops.
- **Per genre x era:** at most 3 films in any cell (Drama 1980s, 2000s; Comedy 1980s, 1990s; Action 1990s). Full table in the CSV.
- **Era: the excluded films are older.** 11 of the 30 (37 %) are from before 1990, against 349 of 2158 (16 %) for the rest (odds ratio 3.0, Fisher exact p = 0.006). In the analysis set it is 8 of 21 against 125 of 1104 (odds ratio 4.8, p = 0.002). So the exclusion removes proportionally more old films. This is partly compensated by cutting quartiles within era, but it does mean the results apply less to older films, which are already few (pre-1980: 52 films left).
- **Genre:** with 21 films the comparison has little power. Comedy (43 % vs 31 %) and Sci-Fi (19 % vs 12 %) are somewhat more frequent among the excluded, Thriller (5 % vs 19 %) and Mystery (0 vs 13 %) less. None of this is tested; it is a description.
- **The 500-token threshold is a first pick**, not a tuned value. `data_check/threshold_check.py` (`threshold_table.csv`, `threshold_hist.png`): of 2188 films, 10 have 500 to 1999 tokens in the dialogue layer and 6 in the description layer. There are no films between 250 and 499 tokens in either layer, and the values below 500 stop at 179 (dialogue) and 170 (description) while the next film has 646 and 755. So 500 sits in an empty stretch (about 180 to 650), not in a dense part of the distribution. The bulk of the films has thousands of tokens (median about 9,000 in dialogue, 12,400 in description). Above 650 the left tail thins out gradually, with no second gap, so a threshold of 1000 would just remove 3 more films (2 dialogue, 1 description) from a continuum. Robustness check for later: repeat the analysis with 250 and 1000. Note that 250 gives exactly the same exclusion as 500 (no film in between), so only 1000 is a real variant unless we try something higher.
- **Limit:** results apply to films with measurable dialogue and description. All Is Lost and The Red Turtle, probably films with genuinely almost no dialogue, are removed by the rule, not because their data is wrong.


### A4. The shared rule, two universes, two era binnings

From `genre_counts/shared_rule_check.py` (`shared_rule_check.csv`), which rebuilds the set independently and was cross-checked: it gives exactly the same 1598 films as the shared table (`features/film_analysis_set.csv`). Only tags, counts, genre, year and the money rule are used; hit and flop are group sizes from the quartile cut, never outcomes. Hits = flops in every cell by construction (middle half dropped). A genre x era cell with fewer than 8 films is excluded.

**Universe (a): 100k+ vote tier** (`movies_features.csv`, 2764 films, 1181 with a script) - now the robustness check.
**Universe (b): all of `movies_features_10k.csv`** (12,673 films, 2020 with a script; the 100k+ tier plus the 10k-100k tier) - the decided universe.

| | (a) 100k+ | (b) all 10k+ |
|---|---|---|
| Films with a script | 1181 | 2020 |
| Superseded rule keeps (any flag, different currency, missing, non-positive dropped) | 1125 | 1730 |
| Shared rule keeps (including the budget-flag exclusion) | 1061 | 1621 |
| First reason dropped: no budget or gross | 28 | 201 |
| ... not in USD | 6 | 31 |
| ... gross flagged rentals, regional or direct-to-video | 19 | 60 |
| ... streaming distributor | 28 | 40 |
| ... released before 1975 | 34 | 59 |
| ... `net_gross` flag (extra) | 5 | 7 |
| ... budget carries another flag (extra) | 0 | 1 |
| Kept by the superseded rule, dropped by the shared rule | 64 (34 before 1975, 28 streaming, 2 non-USD) | 109 |
| Dropped by the layer rule (fewer than 500 tokens) | 19 | 23 |
| **Final N** | **1042** | **1598** |
| (superseded rule + layer rule) | 1104 | 1705 |

**Hit and flop group sizes (hits = flops), current era bins** (pre-1980 is now 1975-79 only, because the shared rule drops earlier films). A genre needs at least 50 per group (the cell minimum removes the pairs shown):

| Genre | (a) hits = flops | (a) pairs removed | (b) hits = flops | (b) pairs removed |
|---|---|---|---|---|
| Drama | 135 | 0 | 223 | 0 |
| Comedy | 83 | 2 | 136 | 5 |
| Action | 81 | 2 | 108 | 3 |
| Adventure | 69 | 5 | 84 | 7 |
| Crime | 54 | 2 | 85 | 3 |
| Thriller | **50** | 3 | 74 | 6 |
| Romance | 30 | 7 | 59 | 3 |
| Horror | 33 | 5 | 52 | 0 |
| Mystery | 33 | 6 | 50 | 3 |
| Sci-Fi | 34 | 5 | 41 | 6 |

- No main genre falls below 50 in either universe. Under (a) Thriller is exactly 50 and no other genre reaches 50. Under (b) Romance (59), Horror (52) and Mystery (50, exactly at the limit) also reach 50.
- With the era bins 1975-1989 merged (not chosen) the (b) sizes would be Drama 222, Comedy 137, Action 109, Adventure 86, Crime 86, Thriller 75, Romance 60, Horror 52, Mystery 51.

**Era bins** (distinct films per bin; genre x era cells below 8 films), decided: current bins.

| | (a) current bins | (a) 1975-1989 merged | (b) current bins | (b) 1975-1989 merged |
|---|---|---|---|---|
| Films in the oldest bin | 20 (1975-79) | 93 | 29 | 179 |
| Genre x era cells, all genres | 109 | 95 | 113 | 97 |
| ... below 8 films | 47 | 34 | 43 | 29 |
| Film-genre pairs removed by that | 133 | 110 | 158 | 124 |
| Main-six cells | 36 | 30 | 36 | 30 |
| ... below 8 films | 5 (all pre-1980: Action 2, Adventure 5, Comedy 2, Crime 2, Thriller 3) | 0 | 5 (all pre-1980) | 0 |

**Script tagging in the less popular films** (tags and counts only; tiny = fewer than 500 tokens in dialogue or description):

| Tier | Films with a script | Tiny | Rate | Dialogue-poor / description-poor | With usable money: films, tiny, rate |
|---|---|---|---|---|---|
| 100k+ | 1181 | 22 | 1.9 % | 13 / 9 | 1061, 19, 1.8 % |
| 10k-100k (the extra films) | 839 | 6 | 0.7 % | 5 / 1 | 560, 4, 0.7 % |

The tiny-layer rate is not higher in the less popular films; if anything it is lower (Fisher exact p = 0.033 for all scripted films, 0.120 among those with usable money). The extra films are somewhat older (20 % of the scripted 10k-100k films are from before 1990, against 13 % at 100k+). The tagging of the extra films does not look worse by this measure; it says nothing about subtler leakage, which only the manual check can look at. The final set under (b) has 556 more films than (a).


### A5. Tier diagnostic (counts only)

From `genre_counts/tier_diagnostic.py` (`tier_diagnostic.csv`). Final set (1598 films, 1042 at 100k+ = 65.2 %, 556 at 10k-100k), labels as decided (quartiles within genre x current era cell, cells under 8 excluded, a film counted once per genre, so the pooled lines count film-genre pairs). Share of 100k+ films:

| | Hits | Flops | Middle | All films in cells |
|---|---|---|---|---|
| Drama | 184/223 (82.5 %) | 63/223 (28.3 %) | 289/436 (66.3 %) | 536/882 (60.8 %) |
| Comedy | 119/136 (87.5 %) | 42/136 (30.9 %) | 163/264 (61.7 %) | 324/536 (60.4 %) |
| Action | 104/108 (96.3 %) | 45/108 (41.7 %) | 168/210 (80.0 %) | 317/426 (74.4 %) |
| Adventure | 81/84 (96.4 %) | 48/84 (57.1 %) | 141/163 (86.5 %) | 270/331 (81.6 %) |
| Crime | 74/85 (87.1 %) | 22/85 (25.9 %) | 111/165 (67.3 %) | 207/335 (61.8 %) |
| Thriller | 61/74 (82.4 %) | 27/74 (36.5 %) | 107/143 (74.8 %) | 195/291 (67.0 %) |
| Pooled, six main genres | 623/710 (87.7 %) | 247/710 (34.8 %) | 979/1381 (70.9 %) | 1849/2801 (66.0 %) |
| Pooled, all genres | 911/1036 (87.9 %) | 377/1036 (36.4 %) | 1387/1971 (70.4 %) | 2675/4043 (66.2 %) |

Decided in D10: no control in the label; the limitation is stated.


### A6. Manual check (one annotator)

From `data_check/summarize_sheets.py` (`manual_check_summary.csv`; numbers only, the filled sheets stay in `private/`). The 95 % intervals are Wilson intervals that treat the points as independent; they are not (10 points per film, 25 films), so the real uncertainty is larger. **Limits:** one annotator (no agreement measure, a second annotator was considered and not done), snippets of at most 15 words, 25 films.

**Main sheet, 250 points, 247 judged correct (3 errors).** Per era group and layer, not pooled:

| Era | Layer | Errors / points | 95 % interval |
|---|---|---|---|
| pre-1990 | dialogue | 2 / 40 | 1.4 % to 16.5 % |
| pre-1990 | description | 0 / 40 | 0 to 8.8 % |
| 1990+ | dialogue | 0 / 60 | 0 to 6.0 % |
| 1990+ | description | 1 / 60 | 0.3 % to 8.9 % |
| both | other and parenthetical | 0 / 50 | |

Only one of the three errors is a content error (a description sentence tagged as dialogue, Teen Wolf). The other two are a lone symbol (`*`, a bullet) sitting in a layer; they vanish with punctuation stripping. With so few errors the old-film estimate is weak, so the statement is "no sign of a high error rate in the sample", not "the error is 1 %".

**Red-flag sheet, 68 rows, 16 films** (a hunting list: it says whether a failure mode exists, not how frequent it is):

| Rule | Real problems |
|---|---|
| `heading_in_dialogue` | 7 of 7 (scene headings inside dialogue), in 4 films |
| `name_colon_in_description` | 1 of 27 (a transition such as "CUT TO:") |
| `caps_run_in_dialogue` | 0 of 10 (shouted speech, legitimate) |
| `quote_in_description` | 0 of 24 for the rule itself; 4 rows are a different problem, garbled text (2 films) |

**Tiny-layer films** (separate sample, 10 random elements per film, never mixed into the rates above):

| Film type | Films | Other layer's content found in this layer: yes / no / unclear |
|---|---|---|
| Dialogue-poor (dialogue inside the description?) | 5 | 10 / 38 / 2 |
| Description-poor (description inside the dialogue?) | 3 | 7 / 19 / 4 |

By film (yes of 10): dialogue-poor 5, 3, 2, 0, 0; description-poor 4, 3, 0. So the hypothesis that dialogue sits untagged inside the description holds in 3 of the 5 dialogue-poor films and not in the other 2 (there the dialogue is simply missing or genuinely absent). **This does not change the exclusion rule** (D9): either a layer is missing or its content is doubtful.


### A7. Prevalence of two failure modes in the final set (1598 films)

From `data_check/prevalence.py` (`prevalence_summary.csv`, `prevalence_by_film.csv`); only tags and characters, no outcome.

- **Scene headings (INT./EXT.) inside dialogue elements:** 1418 of 1,440,867 dialogue elements (0.10 %), in 177 films (11.1 %). In 37 films they are more than 1 % of the dialogue elements. The failure mode is real but rare per element.
- **Symbols that are not prose** (share of whitespace tokens with a symbol that is not a letter, a digit or ordinary punctuation; accented letters do not count): median 0.12 %, 95th percentile 2.1 %, maximum 14.3 %. 150 films are above 1 %, 83 above 2 %, 25 above 5 %, 3 above 10 %. It is a continuum, without a gap, so **it gives no clean exclusion rule**.
- **The measure mixes two different problems.** Old scanned scripts (The Fly II, Fatal Attraction, Rocky II, Mallrats, Con Air) carry scan symbols (bullets, tildes, backslashes). Modern scripts (22 Jump Street, Minari, The Farewell) carry markup residue (`\*`, `[` `]`) and are not corrupted words. A first version counted accented letters and wrongly flagged a French-language film (Anatomy of a Fall); it was fixed to count symbols only.
- **The measure does not isolate the garbled-text films.** The two films the manual check called "text seems encrypted" are rank 1 (The Fly II, 14.3 %) and rank 113 (Basic Instinct 2, 1.35 %) of 1598. So rank 113 is not separated from the rest by this measure. A measure for corrupted words (letters inside words, not symbols) would need a second method.
- **What this means for D17 and D18.** Stripping non-letter symbols in the preprocessing removes the symbol type of noise in both cases; it does not repair corrupted words. Headings inside dialogue are rare per element (0.1 %), and 37 films are above 1 %; decided in D17: no cleaning and no sensitivity run, the limitation is stated.


### A8. Superseded numbers (kept for the trace)

These numbers come from the first money rule (any infobox flag, different currencies dropped) and were the basis of the first manual-check sheets. They are replaced by the shared rule (D6). The first manual-check sheets in `data_check/` were drawn from this set and were redrawn on 2026-11-19 from the final set (the earlier versions are in git history; none had been filled).

- Films with a script, budget and gross (same currency, positive): 1149 before flags; 25 flagged films excluded gave 1125; 4 films with different budget and gross currency had already been dropped.
- 30 films with an empty or tiny layer (fewer than 500 tokens); 21 were in the 1125 set, so **the final set was 1104 films**.
- Group sizes on the 1104 (current bins, minimum cell 8): Drama 147, Comedy 87, Action 82, Adventure 74, Crime 58, Thriller 54 hits and flops. 93 % of the films had more than one genre (1030 of 1104).
- Who was excluded by the layer rule: 11 of the 30 from before 1990 (37 %) against 349 of 2158 (16 %) for the rest (Fisher p = 0.006); in the 1104-set context 8 of 21 against 125 of 1104 (p = 0.002). Comedy and Sci-Fi were somewhat more frequent among the excluded, Thriller and Mystery less (21 films, not tested).
- Pre-1980 bin: 52 films, of which 32 were released before 1975 and are dropped by the shared rule, leaving 20.
- Hits and flops for the six genres under (b) and (c) multi-genre treatments (`cells_by_treatment.csv`) were computed on this set.


## 7. Rules for this folder

- Screenplays are copyrighted (MovieSum is CC BY-NC 4.0). No script text is committed. Anything containing script text goes in `analysis/screenplays/private/` (gitignored).
- Snippets shown in chat are at most 15 words.
