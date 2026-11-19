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
- The manual-check sheets (redrawn from the final set) are not yet filled by Angi.

## Open decisions (Angi decides)

1. **Genre(s)** and the rule for choosing them. Not decided. Multi-genre treatment is decided (see above), so a genre means "film tagged X". The rule must be written before looking at any text result.
   - **Facts.** With the 50-film threshold, the era rule and the shared money rule, on the decided universe (b) the genres with at least 50 hits and 50 flops are Drama (223), Comedy (136), Action (108), Adventure (84), Crime (85), Thriller (74), Romance (59), Horror (52) and Mystery (50, exactly at the limit). About 93 % of films had more than one genre on the superseded 1104-film set, so results for, say, Drama and Crime are not independent (`genre_overlap.csv`, computed on that set; to be recomputed on the final set).
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
19. ~~Universe~~ Decided: all films of the 10k+ table (b); 100k+ only is the robustness check (see above).
20. ~~Era bins~~ Decided: current bins, minimum cell 8 (see above).
21. ~~Tier control~~ Decided: no control in the hit/flop definition; limitation stated, tier-only baseline, 100k+ robustness check (see above).

## Questions the vectors cannot answer

- Which words drive a difference. That is Scattertext's job.
- Profit. The vector alone says nothing about it.

## Rules for this folder

- Screenplays are copyrighted (MovieSum is CC BY-NC 4.0). No script text is committed. Anything containing script text goes in `scripts-analysis/private/` (gitignored).
- Snippets shown in chat are at most 15 words.
