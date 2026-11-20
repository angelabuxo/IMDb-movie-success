# Decisions: screenplay analysis (Q3)

## Context

Group project (DTU 02805). Main question: what best predicts whether a movie earns a lot compared with its budget: its genre, the people who make it, or its script?

My part is Q3: within one genre, do hit scripts differ from flop scripts?

Hand-in: Fri 27 Nov 2026.

## What Word2Vec gives us

- **One vector per word, not per script.** Word2Vec slides over every token position in the training text and learns to predict neighbouring words (Skip-gram, negative sampling). The vectors are a by-product. Words in similar contexts get similar directions.
- **Unsupervised.** It knows nothing about hit/flop or money. Profitability enters only when we compare script vectors with labels.
- **Static.** One vector per word type, so no polysemy. The axes have no individual meaning.
- **A script vector is built by us**, as the mean of the vectors of all word occurrences in the script. It captures what kind of contexts the vocabulary comes from.
- **What the script vector loses:** word order, plot, arc, rare words (diluted by the mean), polysemy. It does NOT reliably capture sentiment (hero and villain have high cosine).
- **Cosine of two vectors** means similar contexts. It does not mean shared vocabulary or same meaning.
- **A plain mean is dominated by frequent words.** Toy example: adding "the" three times to an action and a romance script moved their cosine from 0.33 to 0.75. So aggregation is a decision (open decision 6).

## Decided (provisional, can be revisited)

| Decision | Reason |
|---|---|
| Represent each script as a Word2Vec-based vector and ask how much it tells about profitability. Compare against baselines. | A vector is only informative relative to simpler alternatives. |
| Analyse dialogue and scene description separately first. The combined text is a third, comparison analysis. | They are different registers. Mixing them hides what drives a result and brings in the dialogue/description proportion, which is close to a genre variable. |
| Leaning (not final): train one Word2Vec on all text, then build two vectors per script (dialogue mean, description mean). | Both vectors then live in the same space and are comparable. |
| One primary analysis fixed before seeing results. Everything else is labelled exploratory. | Many variants means some look significant by luck. |
| Write the hypothesis before seeing results. | Avoids fitting the story to the outcome. |
| Checks: shuffle the hit/flop labels (null); TF-IDF as baseline representation; genre-only as baseline; nearest-neighbour sanity check of a few words; hand validation of the layer split (Task 2). | Each one tests a different way the result could be fake or trivial. |
| MovieSum tags -> layers: `dialogue` -> dialogue, `scene_description` -> description, `parenthetical` -> its own layer, `character` and `stage_direction` -> other. The parenthetical layer is excluded from both main layers for now. | Parentheticals like "(exasperated)" are not spoken, so adding them to dialogue would contaminate "what the audience hears". Whether to add them later is open (see below). |
| No cleaning rules in the parser: every element stays in the layer its tag says. Junk (non-header `stage_direction`, numeric or empty `character`, empty tags) is counted as possible leakage in the statistics. | `character` and `stage_direction` are outside both analysis layers, so this noise does not enter the vectors. A filter rule would be one more thing to defend. |
| Layer statistics (data check) use all 2188 unique ids (one record per id, see the duplicates rule), pooled over train/val/test. The manual sample of 25 films comes from the films with budget and gross. | The statistics check the parser, which does not depend on budget. The sample should be films that reach the analysis. The splits are for summarisation, so they are irrelevant here. |
| Duplicated imdb ids (12): keep the record with the longest script (most characters). The `* 2.jsonl` copies are ignored (identical to the originals). | The ids are different versions of the same film, so one has to be chosen. Longest = most complete. Applied in `parse_layers.one_per_id`. |
| Hit/flop is relative within the genre (top quarter vs bottom quarter of ln(gross/budget)), not the absolute "gross >= 2x budget". The middle half is dropped. | The absolute rule gives almost no flops in this sample (e.g. Action: 248 hits, 18 flops) and answers a different question (recovering a multiple). We tried it and it does not discriminate here. Dropping the middle leaves half of each genre's films. |
| **One shared cleaning rule for money: `analysis.clean_rule` (called, never rewritten) AND no `net_gross` flag in the budget or gross flag column AND no flag on the budget.** A film that passes is `usable_money`. The rule requires budget and gross in US dollars and at least $1,000, drops grosses flagged rentals, regional or direct-to-video, drops films of a streaming distributor, and keeps only films released 1975 to 2025. Non-USD films follow the rule and are dropped (2 of the superseded 1104). A film with any flag on the budget is excluded by the same principle (1 film in the 10k+ scripted set). | **The principle (decision 3 extended):** a gross that is not comparable with a worldwide theatrical gross makes a film look like a flop through measurement, not performance. That applies to flagged infobox grosses (already excluded), to films before 1975 (the repo's README says their grosses are mostly distributor rentals; **we have not checked that claim**), and to streaming-distributor films (28 films in the 100k+ scripted set, all from 2016 on; 14 of the 76 films of the 2020s bin). **The rule was chosen on this principle and NOT by looking at hits or flops; no outcome was used to justify it.** It replaces the earlier rule that excluded any infobox flag (25 films) and replaces the earlier "two rules" wording. The group will need a single rule for all three questions; this is the proposal. |
| Minimum 50 films in each of the hit and flop groups for a genre to be a main result. Genres below that are exploratory or out. | 30 is a power threshold, not a validity one. With small groups a null result is not informative: there may be a difference we cannot see. The label shuffle protects against false positives, not against low power. |
| Era rule, written before seeing the decade table: if the share of hits under the within-genre cut is roughly flat across decades, keep that cut. If not, cut quartiles within genre x decade. | If older films have higher gross/budget, hits are mostly old and flops recent, and Word2Vec or Scattertext would find decade differences (vocabulary, script format) and we would read them as success. Era is the most dangerous confounder in this part. |
| **Universe: all films of `movies_features_10k.csv` with a MovieSum script**, the shared money rule and the layer rule (final N 1598). The 100k+ vote tier alone (final N 1042) becomes a robustness check: repeat the analysis on the 100k+ films only. | (1) Group sizes are comfortable (Thriller is exactly 50 at 100k+, a knife edge). (2) Selection on IMDb votes is partly selection on success, since votes are a consequence of popularity: a flop inside the 100k+ tier is a film with many votes but a low return. The 10k tier reduces this bias **without removing it**: this is a limitation of the whole analysis. (3) It is the universe the other two questions already use, so the common set is much larger. |
| **Era bins stay as decided:** before 1980, 1980s, 1990s, 2000s, 2010s, 2020s, with minimum cell size 8. 1975-79 is NOT merged into the 1980s. | The era effect is the main confounder and the base rate of hits differs strongly between the 1970s and the 1980s, so merging would reintroduce era inside a bin. Cells below 8 are already excluded by the rule and only a few film-genre pairs are lost (158 of 4201 at 10k+). |
| **The shared files are built** by `scripts/imdb/build_analysis_set.py` and shared as a dated bundle (`analysis-set-2026-11-19.tar.zst`, branch `angi/shared-analysis-set`, not yet committed): `film_analysis_set.csv` (every film of the 10k+ table, with `tier`, `clean_team_rule`, the net_gross flag, `usable_money`, `usable_ratio_same_currency`, `in_moviesum`, `in_common_set` and the exclusion reasons), `film_genres.csv` and a data dictionary (also `docs/analysis-set.md`). `in_common_set` is the set on which all three questions are compared head to head; each question can use the wider set for its own analysis. `usable_ratio_same_currency` is `usable_money` with the US-dollar condition replaced by "budget and gross in the same currency" (6953 films against 6346: 607 more, 561 of them in rupees), for the questions that only need the ratio; `usable_money` and `in_common_set` (Q3's set) are unchanged. Budgets in different currencies are not comparable in absolute amounts, so an analysis that compares films of similar budget needs USD. No hit/flop labels, no token counts, no script text. | The group asked for the decisions in shared data so that the old analyses can be re-run on the same table. Making the sets a single filter keeps Q3 from being enlarged silently. |
| **Tier control: no control for the vote tier in the definition of hit/flop** (confirmed by Angi). (i) State the limitation: hits are 82-96 % 100k+ films and flops 26-57 %, and votes are partly a consequence of success, so conditioning on tier would condition on a consequence of the outcome. (ii) Add a "tier-only" baseline (predict hit/flop from tier, genre and era alone) to the comparisons: the script vector has to do better than that to say anything about the script. (iii) Repeat the main analysis on the 100k+ films only, as a robustness check. | Cutting quartiles within genre x era x tier is not viable: too few films per group. Evidence: the tier diagnostic above. |
| **Multi-genre films: a film counts in every genre it has (treatment (a)).** "Genre X" therefore means "film tagged X". Results for different genres are not independent: whenever more than one genre is analysed, the overlap must be reported (e.g. Comedy-Action share 55 films, Action-Adventure 154; `genre_counts/genre_overlap.csv`). The genre choice itself stays open. | Rejected (b), only the first-listed genre: the `genres` column is in alphabetical order (all 2592 multi-genre rows), so "first" means "first in the alphabet", not "main". Thriller gets 0 films, Drama 48 (below 50), and the groups are not comparable kinds of film. Rejected (c), only single-genre films: only 74 films qualify and Action has none. |
| **Exclude the films whose dialogue layer or scene-description layer is empty or tiny**, from the whole analysis (both layers, not layer by layer). Rule, in `data_check/layer_stats.py`: fewer than 500 whitespace tokens in the dialogue layer or in the description layer (`tiny_layer`), counted on non-empty `<dialogue>` and `<scene_description>` elements of the script kept by the duplicate rule. The exclusion is applied before the hit/flop quartiles are cut, so hits and flops stay balanced in each genre x era cell. The 500 threshold was my first choice in the data check, not a tuned value. | (1) A script vector is the mean of the word vectors in a layer. With no tokens there is no mean, so these films cannot be represented in that layer. (2) The dialogue-vs-description comparison only makes sense on the same films. If each layer used different films, a difference between layers could come from the films and not from the layer. (3) Among the 18 films with almost no tagged dialogue, the description layer is about twice as long as normal (median about 26,700 tokens vs about 12,400). One likely explanation is dialogue sitting untagged inside the description layer, which would make the description vector unreliable too. This is a hypothesis built from counts and was not verified by reading these films, and the exclusion does not depend on it: either a layer is missing or its content is doubtful. (4) The 12 films with an empty or tiny description have hundreds of `character` elements but 0 or 1 `scene_description` and many empty `<scene />` tags, so their scene content was lost or never tagged. (5) The rule uses only the structure of the scripts, never hit/flop, budget or gross, so it cannot select on the outcome. |
| Era: hit/flop quartiles are cut within genre x era cells (definition (iii)). Evidence: `genre_counts/decade_hits.csv` (table below). | The hit share depends strongly on the decade (about 70 % in the 1970s, 12-14 % in the 2000s and 2020s, when 25 % is expected everywhere) and the pattern repeats in all six large genres. Cutting inside era removes era from the hit/flop contrast by construction, so era-specific vocabulary and script formats cannot be read as success differences. No shuffle null is needed for the decade spread: the differences are far too large. |
| Era bins (current, open decision 20): before 1980, 1980s, 1990s, 2000s, 2010s, 2020s. **Minimum cell size 8 films** (genre x era). Cells below 8 are excluded from the primary analysis and listed. This replaces the earlier "cells under 4 are dropped" rule. | Very few early scripts, and a quartile on 4 to 7 films gives groups of 1 or 2 films. |
| **Manual check, main sample:** 25 films drawn at random (fixed seed 2805) from the final set (1598 films: 179 before 1990, 1419 from 1990 on; drawn 2026-11-19), stratified by era: 10 from before 1990 and 15 from 1990 on. Error rates are reported per era group (pre-1990, 1990+) and per layer, and are NOT pooled into one percentage. | The 30 excluded films are concentrated before 1990 (37 % against 16 %), which suggests that tagging quality depends on era, so errors may be more frequent in older films that pass the threshold. A plain random sample would contain only about 3 pre-1990 films. The sample over-represents old films (10 of 25 against about 11 % of the superseded set), so a pooled rate would not describe the full set. |
| **Separate small sample of the tiny-layer films of the final universe** (23 films with a script and usable money but a layer under 500 tokens: 13 dialogue-poor, 10 description-poor), never mixed into the error rates above: 5 dialogue-poor films and 3 description-poor films, 10 random elements each, from the description layer (dialogue-poor films) or the dialogue layer (description-poor films). Same 15-word limit, text only in `private/`. A column records "dialogue inside description: yes/no/unclear". | Tests directly the hypothesis that dialogue sits untagged inside the description layer in the dialogue-poor films. |
| Manual-check points, stratified by layer, about 10 per film: 4 dialogue, 4 scene description, 2 from other (parenthetical, stage_direction, character). Each point is one element, at most 15 words, the layer the parser gave it, and the tags (not the text) of the previous and next element. Only this stratified sample is used to report error rates; no other sample is mixed into them. | `character` and `dialogue` elements outnumber `scene_description` about two to one, so a plain random sample gives too few description points to estimate an error rate per layer. |
| A separate targeted red-flag sheet (exploratory, no error rate): an automatic list of suspicious elements (description with quote marks or an ALL-CAPS name followed by a colon; dialogue with INT./EXT. or long all-caps runs), about 5 per film for the same 25 films. | Hunts for rare failure modes that a random sample would miss. |
| **Headings (INT./EXT.) inside dialogue: nothing is cleaned in the parser, and no sensitivity analysis is run** (decided by Angi, 2026-11-19). | They are 0.10 % of dialogue elements (1418 of 1,440,867), in 177 of 1598 films; in 37 films they exceed 1 %. The parser keeps its rule "every element stays in the layer its tag says". The limitation is stated: 11 % of films have at least one such element in the dialogue layer. |
| **Symbols that are not letters are removed in the preprocessing** (decided by Angi, 2026-11-19). Corrupted words are not repaired and no film is excluded for garbled text. | Removes scan symbols (bullets, tildes, backslashes) and markup residue (`\*`, `[` `]`). The measure of symbol share has no gap (150 films above 1 %, a continuum) and does not isolate the two films called garbled in the manual check (ranks 1 and 113 of 1598), so it cannot support an exclusion rule. Rare tokens from corrupted words are dropped later by a minimum word count (value open decision 15). |
| **Genres (open decision 1, option 1):** primary **Comedy and Action**, each analysed in the dialogue layer and in the description layer: 4 tests, **all four are reported, never only the best**. **Drama is an exploratory check.** The other genres are not analysed. The hypothesis is written before any text result (open decision 10). | Hypothesis-driven choice with enough films per group (hits = flops: Comedy 136, Action 108; Drama 223). Drama is the largest genre, so it serves as a check with maximum power. Results for different genres are not independent (a film counts in every genre it has); the overlap is reported. |
| **Hypothesis and measure (open decision 10), written before any text result.** Comedy: the difference between hits and flops is larger in the dialogue layer than in the description layer. Action: it is larger in the description layer. Measure: AUC in cross-validation of hit versus flop from the script vector, per genre and layer; the quantity compared is AUC(dialogue) minus AUC(description). Each AUC is compared with (a) the baseline "tier + genre + era", (b) TF-IDF as a representation and (c) the label shuffle. All 4 tests are reported, nulls included. Drama is exploratory. | The conjecture (not a result): humour lives in how people speak, action spectacle in what is described. A vector that does no better than the tier baseline says nothing about the script. A shuffle protects against false positives, not against low power (minimum 50 films per group). |
| **Layers (open decision 5):** dialogue and description are tested separately (the 4 tests); "both together" is only a comparison. | Mixing them hides what drives a result and brings in the dialogue/description proportion, which is close to a genre variable. |
| **Training (open decisions 14, 8):** Word2Vec is trained on the 2188 unique scripts in one space for both layers, our own corpus. GloVe is an optional robustness check. | No label is used, so no outcome leaks into the vectors; the test films do take part in the training, which is stated as a limitation. Size: 48.3 million whitespace tokens (20.2 M dialogue, 28.2 M description; part of the description count is punctuation split off), measured in `layer_stats.csv`. A pretrained space describes its own corpus (Wikipedia, news), not screenplays. |
| **Word2Vec settings (open decisions 7, 15):** skip-gram with negative sampling, 100 dimensions, window 5, `min_count` 20, 5 seeds with `workers=1`. Robustness: windows 2 and 10. | A narrow window gives more syntactic information, a wide one more topical. Small corpus, so no more dimensions. `workers=1` makes a run reproducible. Seeds are reported by mean and spread of the AUC; **vectors of different seeds are never mixed** (each space has its own axes). `min_count` 20 drops rare words, and with them corrupted words, but also some legitimate rare vocabulary (limitation). |
| **Aggregation (open decision 6):** TF-IDF-weighted mean as primary, plain mean as robustness. The IDF is computed over the 2188 scripts, **per layer** (a document is the layer of one film). | The plain mean is dominated by frequent words (the "the" example above). IDF = ln(N/df) gives weight 0 to words in every script. Per layer so the documents are comparable within a layer. |
| **Preprocessing (open decisions 9, 11, 12, 13, 16):** lowercase; symbols that are not letters removed; no lemmatisation; stopwords kept (Word2Vec needs the context, the IDF weighting reduces them); one tokeniser for both layers that joins detached contractions ("do n't" to "don't") and removes attached punctuation; each element (one line of dialogue or one description) is its own context unit and the window never crosses elements; the film's own character names (from its `<character>` tags) removed **only when the token is a character-label token in fewer than 20 of the 2188 films** (`NAME_MAX_FILMS` = 20, about 1 % of the films); parentheticals out of both layers. | A different tokenisation per layer could make a "difference between layers" an artefact. **Why the name filter:** removing every label token took 10.8 % of the dialogue tokens and 17.3 % of the description tokens, because labels are often ordinary words (`voice` is a label in 1419 films, `man` in 1264, `the` in 1112, `guard`, `officer`, `doctor` in 400-580). That deletes the word wherever the label is used and unequally across layers, which is the kind of artefact the single tokeniser was meant to avoid. With the filter it is 1.47 % and 4.2 % (1305 of 34,020 label token types are kept as common words); names shared by many films are reduced by the IDF. **The value 20 is arbitrary, there is no natural cut** (names and ordinary words mix: `john` is a label in 233 films, so it stays in, `police` and `door` in 100 to 300). It was fixed before any AUC was computed, and the only criterion is "the token is specific to a few films"; it is not to be changed after seeing results. Numbers: `preprocess/preprocess_summary.csv`. Implementation choices that are mine, to review: digits are removed too (they are not letters), hyphenated words are split, an apostrophe inside a word is kept. |
| **Franchises (open decision 4):** only a robustness check, repeating the analysis without the films flagged as franchise, using the raw flag, which is noisy; stated as a limitation. | The Wikidata series property is not a clean franchise flag. |
| **Limitations to state in the paper (decided):** a character name that is also an ordinary word and a label in fewer than 20 films (e.g. Will, Grace) is removed from that film's text, while common first names shared by 20 or more films stay in; the cut of 20 is arbitrary; label tokens such as `int`, `ext` and `cut` show that some headings are tagged as characters; the test films take part in the unsupervised training; `min_count` 20 also drops some legitimate rare vocabulary. | They follow from decisions above and do not change them. |

### Test settings fixed before any AUC (Gyula, 2026-11-20, for Angi to review)

The table above leaves these unset. They were written and committed before any AUC was computed; `word2vec/` implements them and they are not to be changed after seeing results.

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

### Evidence: share of hits per decade, quartiles cut within genre only (definition (ii))

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

### Evidence: the 30 films with an empty or tiny layer

From `data_check/tiny_layers.csv` (tags and counts only, no text). A layer is "tiny" below 500 tokens. 21 of the 30 were in the superseded 1104-film set; 19 of them in the 100k+ set under the shared rule (see below).

- **Description tiny or empty (12 films, e.g. Gandhi, Wild Wild West, The Next Three Days).** The film has hundreds of `character` elements, so the dialogue is there, but there are zero or one `scene_description` elements, and often many empty `<scene />` markers (up to 186). The scene content was probably lost or never tagged.
- **Dialogue tiny or empty (18 films).** Few or no `character` elements (0 to 9, one case 43) while the description layer is large: median 26,700 tokens against a median of about 12,400 for all films. This suggests that in these films the dialogue sits in the description layer untagged. If so, the description layer is contaminated in the same films, which the name and heading proxies would not catch. Two of them (All Is Lost, The Red Turtle) are known to have almost no dialogue and are probably genuine.
- This is a hypothesis from counts. The manual check is what can confirm it.

### Cost of the exclusion and who is excluded (numbers on the superseded 1125 / 1104 films)

From `genre_counts/tiny_vs_rest.py` (`tiny_vs_rest.csv`). Only year, genre and the tiny flag are used, never outcomes.

- **Size:** 21 of 1125 films (1.9 %).
- **Per genre (films tagged):** Drama 11, Comedy 9, Action 7, Adventure 3, Crime 3, Thriller 1. After the exclusion the six main genres still have 147, 87, 82, 74, 58 and 54 hits and flops.
- **Per genre x era:** at most 3 films in any cell (Drama 1980s, 2000s; Comedy 1980s, 1990s; Action 1990s). Full table in the CSV.
- **Era: the excluded films are older.** 11 of the 30 (37 %) are from before 1990, against 349 of 2158 (16 %) for the rest (odds ratio 3.0, Fisher exact p = 0.006). In the analysis set it is 8 of 21 against 125 of 1104 (odds ratio 4.8, p = 0.002). So the exclusion removes proportionally more old films. This is partly compensated by cutting quartiles within era, but it does mean the results apply less to older films, which are already few (pre-1980: 52 films left).
- **Genre:** with 21 films the comparison has little power. Comedy (43 % vs 31 %) and Sci-Fi (19 % vs 12 %) are somewhat more frequent among the excluded, Thriller (5 % vs 19 %) and Mystery (0 vs 13 %) less. None of this is tested; it is a description.
- **The 500-token threshold is a first pick**, not a tuned value. `data_check/threshold_check.py` (`threshold_table.csv`, `threshold_hist.png`): of 2188 films, 10 have 500 to 1999 tokens in the dialogue layer and 6 in the description layer. There are no films between 250 and 499 tokens in either layer, and the values below 500 stop at 179 (dialogue) and 170 (description) while the next film has 646 and 755. So 500 sits in an empty stretch (about 180 to 650), not in a dense part of the distribution. The bulk of the films has thousands of tokens (median about 9,000 in dialogue, 12,400 in description). Above 650 the left tail thins out gradually, with no second gap, so a threshold of 1000 would just remove 3 more films (2 dialogue, 1 description) from a continuum. Robustness check for later: repeat the analysis with 250 and 1000. Note that 250 gives exactly the same exclusion as 500 (no film in between), so only 1000 is a real variant unless we try something higher.
- **Limit:** results apply to films with measurable dialogue and description. All Is Lost and The Red Turtle, probably films with genuinely almost no dialogue, are removed by the rule, not because their data is wrong.

### Evidence: the shared rule, two universes, two era binnings

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

### Evidence: tier diagnostic (counts only)

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

Nothing is decided from this; whether it needs a control is open decision 21.

### Evidence: manual check, filled 2026-11-19 (one annotator, Angi)

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

By film (yes of 10): dialogue-poor 5, 3, 2, 0, 0; description-poor 4, 3, 0. So the hypothesis that dialogue sits untagged inside the description holds in 3 of the 5 dialogue-poor films and not in the other 2 (there the dialogue is simply missing or genuinely absent). **This does not change the exclusion rule** (decision "Exclude the films whose dialogue layer...", point 5 and the reasoning): either a layer is missing or its content is doubtful.

### Evidence: prevalence of two failure modes in the final set (1598 films)

From `data_check/prevalence.py` (`prevalence_summary.csv`, `prevalence_by_film.csv`); only tags and characters, no outcome.

- **Scene headings (INT./EXT.) inside dialogue elements:** 1418 of 1,440,867 dialogue elements (0.10 %), in 177 films (11.1 %). In 37 films they are more than 1 % of the dialogue elements. The failure mode is real but rare per element.
- **Symbols that are not prose** (share of whitespace tokens with a symbol that is not a letter, a digit or ordinary punctuation; accented letters do not count): median 0.12 %, 95th percentile 2.1 %, maximum 14.3 %. 150 films are above 1 %, 83 above 2 %, 25 above 5 %, 3 above 10 %. It is a continuum, without a gap, so **it gives no clean exclusion rule**.
- **The measure mixes two different problems.** Old scanned scripts (The Fly II, Fatal Attraction, Rocky II, Mallrats, Con Air) carry scan symbols (bullets, tildes, backslashes). Modern scripts (22 Jump Street, Minari, The Farewell) carry markup residue (`\*`, `[` `]`) and are not corrupted words. A first version counted accented letters and wrongly flagged a French-language film (Anatomy of a Fall); it was fixed to count symbols only.
- **The measure does not isolate the garbled-text films.** The two films the manual check called "text seems encrypted" are rank 1 (The Fly II, 14.3 %) and rank 113 (Basic Instinct 2, 1.35 %) of 1598. So rank 113 is not separated from the rest by this measure. A measure for corrupted words (letters inside words, not symbols) would need a second method.
- **What this means for decisions A and B** (below, open): stripping non-letter symbols in the preprocessing removes the symbol type of noise in both cases; it does not repair corrupted words. Headings inside dialogue are rare per element (0.1 %), and 37 films are above 1 %; decided: no cleaning and no sensitivity run, the limitation is stated.

## Superseded (kept for the trace)

These numbers come from the first money rule (any infobox flag, different currencies dropped) and were the basis of the first manual-check sheets. They are replaced by the shared rule above. The first manual-check sheets in `data_check/` were drawn from this set and were redrawn on 2026-11-19 from the final set (the earlier versions are in git history; none had been filled).

- Films with a script, budget and gross (same currency, positive): 1149 before flags; 25 flagged films excluded gave 1125; 4 films with different budget and gross currency had already been dropped.
- 30 films with an empty or tiny layer (fewer than 500 tokens); 21 were in the 1125 set, so **the final set was 1104 films**.
- Group sizes on the 1104 (current bins, minimum cell 8): Drama 147, Comedy 87, Action 82, Adventure 74, Crime 58, Thriller 54 hits and flops. 93 % of the films had more than one genre (1030 of 1104).
- Who was excluded by the layer rule: 11 of the 30 from before 1990 (37 %) against 349 of 2158 (16 %) for the rest (Fisher p = 0.006); in the 1104-set context 8 of 21 against 125 of 1104 (p = 0.002). Comedy and Sci-Fi were somewhat more frequent among the excluded, Thriller and Mystery less (21 films, not tested).
- Pre-1980 bin: 52 films, of which 32 were released before 1975 and are dropped by the shared rule, leaving 20.
- Hits and flops for the six genres under (b) and (c) multi-genre treatments (`cells_by_treatment.csv`) were computed on this set.

## Pending

- Committed locally on 2026-11-19: the shared files on `angi/shared-analysis-set` (worktree `build/shared-analysis-set`) and this folder on `angi/scripts-analysis`. Not pushed, no PR yet.
- The manual-check sheets were filled by Angi on 2026-11-19 (summary above). `summarize_sheets.py` and `prevalence.py` are written and run; not yet committed.
- Decided after the check: A (nothing cleaned, no sensitivity run), B (non-letter symbols removed in preprocessing), genres (Comedy and Action primary, Drama exploratory). Hypothesis (open decision 10) to be written next.

## Open decisions (Angi decides)

1. ~~Genre(s)~~ Decided: primary Comedy and Action, Drama exploratory (see the decided table); original options kept below for the trace. Multi-genre treatment is decided (see above), so a genre means "film tagged X". The rule must be written before looking at any text result.
   - **Facts.** With the 50-film threshold, the era rule and the shared money rule, on the decided universe (b) the genres with at least 50 hits and 50 flops are Drama (223), Comedy (136), Action (108), Adventure (84), Crime (85), Thriller (74), Romance (59), Horror (52) and Mystery (50, exactly at the limit). About 93 % of films had more than one genre on the superseded 1104-film set, so results for, say, Drama and Crime are not independent (`genre_overlap.csv`, computed on that set; to be recomputed on the final set).
   - **Option 1: Comedy and Action**, tied to a hypothesis written before any text result (Comedy may depend more on dialogue, Action on scene description). Both have many films per group. Two genres x two layers means four tests: report all four, never pick the best.
   - **Option 2: the largest genre (Drama).** Maximum power and a simple rule, but a broad catch-all genre.
   - **Option 3: all genres above the size threshold**, with one declared primary and the rest exploratory.
2. ~~Definition of hit and flop~~ Decided: relative, within genre x era (see above). Note: gross is not profit.
3. ~~The middle~~ Decided: dropped.
4. ~~Franchises and era~~ Decided (see the decided table). not the focus, but report results with and without them as a cheap robustness check.
5. ~~Primary layer~~ Decided (see the decided table). dialogue, description or both.
6. ~~Aggregation~~ Decided (see the decided table). plain mean vs TF-IDF-weighted mean (and stopword handling).
7. ~~Window size~~ Decided (see the decided table). narrow (style/syntax) vs wide (topic).
8. ~~Training data~~ Decided (see the decided table). our scripts vs a pretrained space (GloVe is trained on Wikipedia and news, not screenplays).
9. ~~Preprocessing~~ Decided (see the decided table). lowercase, character names, stopwords, lemmatisation.
10. ~~Hypothesis~~ Decided (see the decided table)., e.g. "description separates hits from flops more than dialogue does".
11. ~~Uniform tokenisation across layers.~~ Decided (see the decided table). The description layer has punctuation split off (` ,`) and the dialogue layer mostly not, and contractions differ (`Do n't` vs `you're`). If not unified, a "difference between layers" could be a tokenisation artefact. Quantified in the data check, not fixed.
12. ~~Context unit.~~ Decided (see the decided table). Treat each element (one dialogue line, one description) as its own context unit, so the Word2Vec window never crosses elements.
13. ~~Character names inside dialogue and description text.~~ Decided (see the decided table). Option: build a per-film name list from that film's own `<character>` tags and drop those tokens.
14. ~~Which scripts train Word2Vec~~ Decided (see the decided table). all 2188 unique ids vs only those with budget and gross.
15. ~~Word2Vec settings~~ Decided (see the decided table). window size, dimensions (window size also under 7).
16. ~~Parentheticals~~ Decided (see the decided table). add them to dialogue later or keep them out.
17. ~~Minimum tokens per layer~~ Decided: films with fewer than 500 tokens in the dialogue or the description layer are excluded (see above).
18. ~~Multi-genre treatment~~ Decided: (a), see above.
19. ~~Universe~~ Decided: all films of the 10k+ table (b); 100k+ only is the robustness check (see above).
20. ~~Era bins~~ Decided: current bins, minimum cell 8 (see above).
21. ~~Tier control~~ Decided: no control in the hit/flop definition; limitation stated, tier-only baseline, 100k+ robustness check (see above).

## Questions the vectors cannot answer

- Which words drive a difference. That is Scattertext's job.
- Profit. The vector alone says nothing about it.

## Rules for this folder

- Screenplays are copyrighted (MovieSum is CC BY-NC 4.0). No script text is committed. Anything containing script text goes in `scripts-analysis/private/` (gitignored).
- Snippets shown in chat are at most 15 words.
