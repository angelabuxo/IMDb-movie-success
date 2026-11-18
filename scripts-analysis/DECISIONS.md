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
| Films whose budget or gross carries an infobox flag (regional, rentals, net_gross, direct_to_video) are excluded (25 films). | A "rentals" or regional-only gross is not comparable with a worldwide gross, so those films would look like flops from measurement error. |
| Minimum 50 films in each of the hit and flop groups for a genre to be a main result. Genres below that are exploratory or out. | 30 is a power threshold, not a validity one. With small groups a null result is not informative: there may be a difference we cannot see. The label shuffle protects against false positives, not against low power. |
| Era rule, written before seeing the decade table: if the share of hits under the within-genre cut is roughly flat across decades, keep that cut. If not, cut quartiles within genre x decade. | If older films have higher gross/budget, hits are mostly old and flops recent, and Word2Vec or Scattertext would find decade differences (vocabulary, script format) and we would read them as success. Era is the most dangerous confounder in this part. |

| **Multi-genre films: a film counts in every genre it has (treatment (a)).** "Genre X" therefore means "film tagged X". Results for different genres are not independent: whenever more than one genre is analysed, the overlap must be reported (e.g. Comedy-Action share 55 films, Action-Adventure 154; `genre_counts/genre_overlap.csv`). The genre choice itself stays open. | Rejected (b), only the first-listed genre: the `genres` column is in alphabetical order (all 2592 multi-genre rows), so "first" means "first in the alphabet", not "main". Thriller gets 0 films, Drama 48 (below 50), and the groups are not comparable kinds of film. Rejected (c), only single-genre films: only 74 films qualify and Action has none. |
| **Exclude the 30 films whose dialogue layer or scene-description layer is empty or tiny**, from the whole analysis (both layers, not layer by layer). Rule, in `data_check/layer_stats.py`: fewer than 500 whitespace tokens in the dialogue layer or in the description layer (`tiny_layer`), counted on non-empty `<dialogue>` and `<scene_description>` elements of the script kept by the duplicate rule. 21 of the 30 are in the 1125-film set, so the **final analysis set is 1104 films**. The exclusion is applied before the hit/flop quartiles are cut, so hits and flops stay balanced in each genre x era cell. The 500 threshold was my first choice in the data check, not a tuned value. | (1) A script vector is the mean of the word vectors in a layer. With no tokens there is no mean, so these films cannot be represented in that layer. (2) The dialogue-vs-description comparison only makes sense on the same films. If each layer used different films, a difference between layers could come from the films and not from the layer. (3) Among the 18 films with almost no tagged dialogue, the description layer is about twice as long as normal (median about 26,700 tokens vs about 12,400). One likely explanation is dialogue sitting untagged inside the description layer, which would make the description vector unreliable too. This is a hypothesis built from counts and was not verified by reading these films, and the exclusion does not depend on it: either a layer is missing or its content is doubtful. (4) The 12 films with an empty or tiny description have hundreds of `character` elements but 0 or 1 `scene_description` and many empty `<scene />` tags, so their scene content was lost or never tagged. (5) The rule uses only the structure of the scripts, never hit/flop, budget or gross, so it cannot select on the outcome. |
| Era: hit/flop quartiles are cut within genre x era cells (definition (iii)). Evidence: `genre_counts/decade_hits.csv` (table below). | The hit share depends strongly on the decade (about 70 % in the 1970s, 12-14 % in the 2000s and 2020s, when 25 % is expected everywhere) and the pattern repeats in all six large genres. Cutting inside era removes era from the hit/flop contrast by construction, so era-specific vocabulary and script formats cannot be read as success differences. No shuffle null is needed for the decade spread: the differences are far too large. |
| Era bins: before 1980, 1980s, 1990s, 2000s, 2010s, 2020s. **Minimum cell size 8 films** (genre x era). Cells below 8 are excluded from the primary analysis and listed. This replaces the earlier "cells under 4 are dropped" rule. | Very few early scripts, and a quartile on 4 to 7 films gives groups of 1 or 2 films. |
| **Manual check, main sample:** 25 films drawn at random (fixed seed 2805) from the 1104 final films, stratified by era: 10 from before 1990 and 15 from 1990 on. Error rates are reported per era group (pre-1990, 1990+) and per layer, and are NOT pooled into one percentage. | The 30 excluded films are concentrated before 1990 (37 % against 16 %), which suggests that tagging quality depends on era, so errors may be more frequent in older films that pass the threshold. A plain random sample would contain only about 3 pre-1990 films. The sample over-represents old films (10 of 25 against 11 % of the set), so a pooled rate would not describe the full set. |
| **Separate small sample of the 30 excluded films**, never mixed into the error rates above: 5 dialogue-poor films and 3 description-poor films, 10 random elements each, from the description layer (dialogue-poor films) or the dialogue layer (description-poor films). Same 15-word limit, text only in `private/`. A column records "dialogue inside description: yes/no/unclear". | Tests directly the hypothesis that dialogue sits untagged inside the description layer in the dialogue-poor films. |
| Manual-check points, stratified by layer, about 10 per film: 4 dialogue, 4 scene description, 2 from other (parenthetical, stage_direction, character). Each point is one element, at most 15 words, the layer the parser gave it, and the tags (not the text) of the previous and next element. Only this stratified sample is used to report error rates; no other sample is mixed into them. | `character` and `dialogue` elements outnumber `scene_description` about two to one, so a plain random sample gives too few description points to estimate an error rate per layer. |
| A separate targeted red-flag sheet (exploratory, no error rate): an automatic list of suspicious elements (description with quote marks or an ALL-CAPS name followed by a colon; dialogue with INT./EXT. or long all-caps runs), about 5 per film for the same 25 films. | Hunts for rare failure modes that a random sample would miss. |

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

(The table was first computed on 1125 films and recomputed on the final 1104; the pattern did not change. 1930s and 1940s have 5 and 4 pairs and are in the CSV.)

### Evidence: the 30 films with an empty or tiny layer

From `data_check/tiny_layers.csv` (tags and counts only, no text). A layer is "tiny" below 500 tokens. 21 of the 30 are in the final analysis set.

- **Description tiny or empty (12 films, e.g. Gandhi, Wild Wild West, The Next Three Days).** The film has hundreds of `character` elements, so the dialogue is there, but there are zero or one `scene_description` elements, and often many empty `<scene />` markers (up to 186). The scene content was probably lost or never tagged.
- **Dialogue tiny or empty (18 films).** Few or no `character` elements (0 to 9, one case 43) while the description layer is large: median 26,700 tokens against a median of about 12,400 for all films. This suggests that in these films the dialogue sits in the description layer untagged. If so, the description layer is contaminated in the same films, which the name and heading proxies would not catch. Two of them (All Is Lost, The Red Turtle) are known to have almost no dialogue and are probably genuine.
- This is a hypothesis from counts. The manual check is what can confirm it.

### Cost of the exclusion and who is excluded

From `genre_counts/tiny_vs_rest.py` (`tiny_vs_rest.csv`). Only year, genre and the tiny flag are used, never outcomes.

- **Size:** 21 of 1125 films (1.9 %).
- **Per genre (films tagged):** Drama 11, Comedy 9, Action 7, Adventure 3, Crime 3, Thriller 1. After the exclusion the six main genres still have 147, 87, 82, 74, 58 and 54 hits and flops.
- **Per genre x era:** at most 3 films in any cell (Drama 1980s, 2000s; Comedy 1980s, 1990s; Action 1990s). Full table in the CSV.
- **Era: the excluded films are older.** 11 of the 30 (37 %) are from before 1990, against 349 of 2158 (16 %) for the rest (odds ratio 3.0, Fisher exact p = 0.006). In the analysis set it is 8 of 21 against 125 of 1104 (odds ratio 4.8, p = 0.002). So the exclusion removes proportionally more old films. This is partly compensated by cutting quartiles within era, but it does mean the results apply less to older films, which are already few (pre-1980: 52 films left).
- **Genre:** with 21 films the comparison has little power. Comedy (43 % vs 31 %) and Sci-Fi (19 % vs 12 %) are somewhat more frequent among the excluded, Thriller (5 % vs 19 %) and Mystery (0 vs 13 %) less. None of this is tested; it is a description.
- **The 500-token threshold is a first pick**, not a tuned value. `data_check/threshold_check.py` (`threshold_table.csv`, `threshold_hist.png`): of 2188 films, 10 have 500 to 1999 tokens in the dialogue layer and 6 in the description layer. There are no films between 250 and 499 tokens in either layer, and the values below 500 stop at 179 (dialogue) and 170 (description) while the next film has 646 and 755. So 500 sits in an empty stretch (about 180 to 650), not in a dense part of the distribution. The bulk of the films has thousands of tokens (median about 9,000 in dialogue, 12,400 in description). Above 650 the left tail thins out gradually, with no second gap, so a threshold of 1000 would just remove 3 more films (2 dialogue, 1 description) from a continuum. Robustness check for later: repeat the analysis with 250 and 1000. Note that 250 gives exactly the same exclusion as 500 (no film in between), so only 1000 is a real variant unless we try something higher.
- **Limit:** results apply to films with measurable dialogue and description. All Is Lost and The Red Turtle, probably films with genuinely almost no dialogue, are removed by the rule, not because their data is wrong.

## Pending

- Nothing waiting on a previous question. The sheets are built (not yet filled by Angi).

## Open decisions (Angi decides)

1. **Genre(s)** and the rule for choosing them. Not decided. Multi-genre treatment is decided (see above), so a genre means "film tagged X". The rule must be written before looking at any text result.
   - **Facts.** With the 50-film threshold and the era rule, six genres have at least 50 hits and 50 flops: Drama (147), Comedy (87), Action (82), Adventure (74), Crime (58), Thriller (54), on the final 1104 films. About 93 % of films (1030 of 1104) have more than one genre, so results for, say, Drama and Crime are not independent (shared films: see `genre_overlap.csv`).
   - **Option 1: Comedy and Action**, tied to a hypothesis written before any text result (Comedy may depend more on dialogue, Action on scene description). Both have many films per group. Two genres x two layers means four tests: report all four, never pick the best.
   - **Option 2: the largest genre (Drama).** Maximum power and a simple rule, but a broad catch-all genre.
   - **Option 3: all genres above the size threshold**, with one declared primary and the rest exploratory.
2. ~~Definition of hit and flop~~ Decided: relative, within genre x era (see above). Note: gross is not profit.
3. ~~The middle~~ Decided: dropped.
4. **Franchises and era:** not the focus, but report results with and without them as a cheap robustness check.
5. **Primary layer:** dialogue, description or both.
6. **Aggregation:** plain mean vs TF-IDF-weighted mean (and stopword handling).
7. **Window size:** narrow (style/syntax) vs wide (topic).
8. **Training data:** our scripts vs a pretrained space (GloVe is trained on Wikipedia and news, not screenplays).
9. **Preprocessing:** lowercase, character names, stopwords, lemmatisation.
10. **Hypothesis**, e.g. "description separates hits from flops more than dialogue does".
11. **Uniform tokenisation across layers.** The description layer has punctuation split off (` ,`) and the dialogue layer mostly not, and contractions differ (`Do n't` vs `you're`). If not unified, a "difference between layers" could be a tokenisation artefact. Quantified in the data check, not fixed.
12. **Context unit.** Treat each element (one dialogue line, one description) as its own context unit, so the Word2Vec window never crosses elements.
13. **Character names inside dialogue and description text.** Option: build a per-film name list from that film's own `<character>` tags and drop those tokens.
14. **Which scripts train Word2Vec:** all 2188 unique ids vs only those with budget and gross.
15. **Word2Vec settings:** window size, dimensions (window size also under 7).
16. **Parentheticals:** add them to dialogue later or keep them out.
17. ~~Minimum tokens per layer~~ Decided: films with fewer than 500 tokens in the dialogue or the description layer are excluded (see above).
18. ~~Multi-genre treatment~~ Decided: (a), see above.

## Questions the vectors cannot answer

- Which words drive a difference. That is Scattertext's job.
- Profit. The vector alone says nothing about it.

## Rules for this folder

- Screenplays are copyrighted (MovieSum is CC BY-NC 4.0). No script text is committed. Anything containing script text goes in `scripts-analysis/private/` (gitignored).
- Snippets shown in chat are at most 15 words.
