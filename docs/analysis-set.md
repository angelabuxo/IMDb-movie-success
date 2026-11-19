# Analysis set: one table of films with the cleaning flags

Built by `scripts/imdb/build_analysis_set.py` from `features/movies_features_10k.csv` (all 12,673 films with 10,000+ IMDb votes)
and the MovieSum screenplays (only to count tokens per layer; no script text is in these files).
Three files: `film_analysis_set.csv` (one row per film), `film_genres.csv` (one row per film-genre pair) and this dictionary.
**No row is deleted.** A film that is excluded says why in a column, so every question can filter its own way.

## Which flag to use

- `usable_money` is the **primary cleaning rule** for money. It is the repo's shared clean rule
  (`analysis/clean_rule.py`, called and not rewritten) **and** no `net_gross` flag **and** no flag on the budget.
- `in_common_set` is the set on which all three questions are compared head to head. Each question can use the wider
  set for its own analysis (for example every film with `usable_money`, with or without a script).
- `usable_ratio_same_currency` is wider than `usable_money`: it also keeps the films whose budget and gross are in the same non-dollar
  currency (for example euros or rupees). Use it for the **ratio** gross/budget (`ln_gross_over_budget`), which has no unit when both
  figures are in the same currency. Every film with `usable_money` also has `usable_ratio_same_currency`.
- `clean_team_rule` is the result of `analysis/clean_rule.py` alone, kept so the two can be compared.

| Column | Keeps | Does not keep |
|---|---|---|
| `clean_team_rule` | films passing `analysis/clean_rule.py`. The rule does not look at `net_gross` or at flags on the budget, so those films pass | non-dollar films, missing figures, amounts under $1,000, grosses flagged rentals-only, regional or direct-to-video, streaming distributors, films outside 1975 to 2025 |
| `usable_money` | `clean_team_rule`, no `net_gross` flag, no flag on the budget; budget and gross in **US dollars** | non-dollar films, and everything `clean_team_rule` drops |
| `usable_ratio_same_currency` | the same conditions as `usable_money`, but budget and gross in the **same currency** (dollars or not) | films with budget and gross in different currencies, and every other exclusion of `usable_money` |
| `in_common_set` | `usable_money` + a MovieSum script + both layers with at least 500 tokens | everything else. Unchanged by `usable_ratio_same_currency` |

## Files

| File | One row per | Rows |
|---|---|---|
| `film_analysis_set.csv` | film | 12,673 |
| `film_genres.csv` | film-genre pair (a film appears once per genre it has) | 32,769 |

## Columns of `film_analysis_set.csv`

| Column | Meaning | Source |
|---|---|---|
| `imdb_id` | IMDb title ID, the key for joining everything | IMDb non-commercial datasets |
| `title` | IMDb primary title | IMDb |
| `year` | release year (IMDb start year) | IMDb |
| `decade` | year rounded down to the decade | derived |
| `era_bin` | before 1980, 1980s, 1990s, 2000s, 2010s, 2020s | derived from `year` |
| `tier` | `100k+` for 100,000+ IMDb votes, `10k-100k` for 10,000 to 99,999 | IMDb votes |
| `genres` | the original IMDb genre string. The genres are in **alphabetical order, not by importance** | IMDb |
| `n_genres` | number of genres | derived |
| `budget`, `gross` | the figures parsed from the Wikipedia infobox, in the currency of `budget_cur` and `gross_cur` (US dollars for every film with `usable_money`) | English Wikipedia infobox (`scripts/imdb/parse_infobox.py`). Not TMDB, not Wikidata |
| `budget_cur`, `gross_cur` | currency code of each figure | the parser |
| `budget_flag`, `gross_flag` | the parser's flags on the figure: `regional`, `rentals`, `net_gross`, `direct_to_video` (blank when none) | the parser |
| `ln_gross_over_budget` | ln(gross / budget). Filled only when both figures are positive and in the same currency. Gross over budget is **not profit**. Use it with `usable_money` | derived |
| `net_gross_flag` | True when the budget or the gross carries a `net_gross` flag | the parser's flags |
| `clean_team_rule` | True when the film passes `analysis/clean_rule.py` | the shared clean rule |
| `usable_money` | `clean_team_rule` and no `net_gross` flag and no flag on the budget. **The primary rule** | see Rules |
| `usable_ratio_same_currency` | `usable_money` with the US-dollar condition replaced by "budget and gross are in the same currency". Use it for the ratio gross/budget only (see Rules) | see Rules |
| `money_exclude_reason` | why `usable_money` is False: the first reason the shared rule reports, then `net_gross flag`, then `budget carries an infobox flag`. Blank when True | derived |
| `in_moviesum` | True when MovieSum has a screenplay for this ID | MovieSum |
| `in_common_set` | True when the film has a script, `usable_money`, and both layers have at least 500 tokens | see Rules |
| `common_set_exclude_reason` | why `in_common_set` is False (`no MovieSum script`, `money: <reason>`, `dialogue or description layer under 500 tokens`). Blank when True | derived |

`film_genres.csv` has two columns, `imdb_id` and `genre`. A film appears once for each genre it has, so "genre X" means
"film tagged X". Results for different genres are not independent: films are in several genres at once.

Hit and flop labels are **not** in these files: they depend on genre x era cells of the set a question uses. Token counts per layer
and script text are not in them either.

## Rules, with the reasons

1. **The money rule (`usable_money`).** `analysis/clean_rule.py` requires budget and gross in US dollars and at least $1,000,
   drops grosses flagged rentals-only, regional or direct-to-video, drops films from a streaming distributor, and keeps only
   films released 1975 to 2025. On top of that, a film with a `net_gross` flag, or with any flag on the budget, is excluded.
   *Reason:* a gross that is not comparable with a worldwide theatrical gross makes a film look like a flop through measurement,
   not performance. That applies to flagged infobox figures, to films before 1975 (the repo's README says their grosses are mostly
   distributor rentals; we have not checked that claim), and to streaming-distributor films. The rule was chosen on this principle and not by looking at hits or flops.
   **Non-dollar films.** `usable_money` needs US dollars, so it drops every film whose figures are in another currency
   (944 films across the whole table, mixed or not). `usable_ratio_same_currency` keeps those whose budget and gross are in the
   *same* currency, because a ratio of two amounts in one currency has no unit. **Budgets in different currencies are not comparable in absolute
   amounts**: an analysis that compares films of similar budget, or uses the budget as a variable, needs US dollars (`usable_money`). The minimum
   amount of 1,000 in the rule is a plain number, so for a non-dollar film it is 1,000 units of its own currency.
2. **The layer rule.** A film is excluded when its dialogue layer or its scene-description layer has fewer than 500 tokens
   (whitespace-separated words of the `<dialogue>` and `<scene_description>` elements of the screenplay). It applies to the whole
   analysis, not layer by layer. *Reasons:* a script vector is the mean of the word vectors of a layer, so with no tokens there is no vector;
   the comparison between layers only makes sense on the same films; in the films with almost no tagged dialogue the description layer is about twice as long as
   normal, which suggests the dialogue sits untagged inside it (a hypothesis from counts, not verified by reading); and the rule uses only the structure of the scripts, never the outcome.
   The threshold 500 is a first pick, not a tuned value.
3. **Duplicated screenplay IDs.** Twelve IDs appear twice in MovieSum with different versions; the record with the most characters is used.
4. **Universe.** All films of the 10k+ table, not only the 100k+ tier. *Reasons:* group sizes are comfortable;
   selection on IMDb votes is partly selection on success (votes follow popularity), and the 10k tier reduces this bias without removing it;
   and it is the universe the network and text questions already use. The 100k+ films alone are a robustness check (`tier`).
5. **Era bins** are before 1980, 1980s, 1990s, 2000s, 2010s, 2020s. The era effect is the main confounder and hit rates differ strongly between the 1970s and the 1980s.

## Counts (exact, from the build)

- Films: 12,673 (`100k+`: 2,764, `10k-100k`: 9,909). With a MovieSum script: 2,020.
- Pass `clean_team_rule`: 6,414 (2,352 at 100k+). Pass `usable_money`: 6,346.
- Pass `usable_ratio_same_currency`: 6,953 (2,360 at 100k+; 1,631 with a MovieSum script), which is `usable_money` plus
  607 films in another currency (INR 561, GBP 21, EUR 17, AUD 3, CAD 2, JPY 2, HKD 1; 35 at 100k+, 572 at 10k-100k).
  Of the 944 films that `usable_money` drops for "not in USD", 607 have budget and gross in the same currency and pass this column; the rest have two different currencies or fail another condition.
- **`in_common_set`: 1,598 films** (1,042 at 100k+, 556 at 10k-100k).

`money_exclude_reason`, all 12,673 films:

| Reason | Films |
|---|---|
| (none: the film passes) | 6,346 |
| no budget or gross parsed | 4,230 |
| budget or gross not in USD | 944 |
| gross flagged rentals, regional or direct-to-video | 514 |
| released before 1975 or after 2025 | 455 |
| streaming distributor | 109 |
| budget carries an infobox flag | 35 |
| net_gross flag | 33 |
| budget or gross under $1,000 | 7 |

`common_set_exclude_reason`, all 12,673 films:

| Reason | Films |
|---|---|
| no MovieSum script | 10,653 |
| (none: the film passes) | 1,598 |
| money: no budget or gross parsed | 201 |
| money: gross flagged rentals, regional or direct-to-video | 60 |
| money: released before 1975 or after 2025 | 59 |
| money: streaming distributor | 40 |
| money: budget or gross not in USD | 31 |
| dialogue or description layer under 500 tokens | 23 |
| money: net_gross flag | 7 |
| money: budget carries an infobox flag | 1 |

### How the two money rules differ

`usable_money` is the shared clean rule with two extra exclusions, so it never keeps a film that `clean_team_rule` drops.
Among the 6,414 films that pass `clean_team_rule`, 68 do not pass `usable_money`
(27 of them at 100k+): 33 because of a `net_gross` flag and 35 because the budget carries another flag.
The repo's old clean rule accepts those films; the new rule does not, for the reason given in Rule 1.
**The paper needs a single rule for all three questions.** `usable_money` is the proposal; the group confirms it.

## Warnings

- **Gross is not profit.** Marketing is not in the budget and cinemas keep part of the gross.
- **Flops are under-represented.** A film needs a budget and a gross in its Wikipedia infobox, which favours films with reported figures.
- **The films removed by the layer rule are concentrated before 1990.** The results apply less to older films.
- **Genre means "film tagged X".** Most films have several genres, so results for different genres are not independent.
- **Selection on votes is partly selection on success.** Films with 10,000+ votes are the films that got an audience.

## Licences and credits

Derived from IMDb's non-commercial datasets (personal and non-commercial use under IMDb's terms), English Wikipedia
(CC BY-SA 4.0; the infobox figures), and MovieSum (CC BY-NC 4.0; Saxena and Keller, Findings of ACL 2024) for `in_moviesum`
and the layer rule. Non-commercial use only. The screenplays are not in these files and remain under their rights holders' copyright.
