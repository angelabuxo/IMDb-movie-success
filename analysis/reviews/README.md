# Reviews analysis — do money, critics and audience agree?

Part of the DTU 02805 final project (*What makes a movie highly profitable?*).  
Notebook: [`01_exploration.ipynb`](01_exploration.ipynb). Helpers live in this folder; derived numbers and figures are under [`outputs/`](outputs/).

## Question

Do theatrical money, critic scores and audience reception move together, and where do they disagree?

## Sample (from data)

| Slice | N | Under budget | Notes |
|---|---:|---:|---|
| Clean 100k+ (all years) | 2 446 | 169 (6.9%) | Wikipedia USD budget/gross |
| Clean 100k+, year ≤ 2017 | 2 095 | 130 (6.2%) | Tag Genome eligibility cut |
| … with Tag Genome reviews | 2 070 | — | Median hundreds of reviews/film |
| Optional 10k–100k (Wikidata money) | 1 126 | 366 (32.5%) | Not hand-checked |
| … year ≤ 2017 | 1 012 | 316 (31.2%) | Flop-rich sensitivity check |

Reviews have **no date** and **no post-2017 coverage**, so every audience-text analysis is restricted to movies released in 2017 or earlier.

**Money proxies.** `log(gross/budget)` (primary) and `gross − budget`. Neither is studio profit: marketing is missing and cinemas keep part of the gross.

## Idea 1 — score agreement

**Measures.** Spearman rank correlation among `log(gross/budget)`, IMDb `averageRating`, Metacritic and Rotten Tomatoes, with a **label-shuffle null** (1 000 permutations).

| Pair | N | ρ | p_null |
|---|---:|---:|---:|
| log(gross/budget) ↔ IMDb | 2 095 | 0.266 | 0.001 |
| log(gross/budget) ↔ Metacritic | 1 834 | 0.261 | 0.001 |
| log(gross/budget) ↔ Rotten Tomatoes | 2 093 | 0.326 | 0.001 |
| IMDb ↔ Metacritic | 1 834 | 0.757 | 0.001 |
| IMDb ↔ Rotten Tomatoes | 2 093 | 0.785 | 0.001 |
| Metacritic ↔ Rotten Tomatoes | 1 833 | 0.931 | 0.001 |

Critics agree strongly with each other and with IMDb; money agrees only weakly with all three. Every observed |ρ| beats the shuffle null.

**Quadrants** (break-even × median IMDb = 7.0): 910 profitable + below-median rating; 63 loved + unprofitable. Extreme examples (from `outputs/`): *Fifty Shades of Grey*, *Epic Movie*, *Twilight: New Moon* on the profitable-low side; *The Boondock Saints*, *Once Upon a Time in America*, *Mr. Nobody* on the loved-unprofitable side.

**What it misses.** The 100k+ cut barely contains flops, so “agreement” is mostly among hits. Optional 10k Wikidata tier (≤2017): ρ(log profit, IMDb) = 0.201 (still above null), with 31.2% under budget.

Figures: `outputs/fig_scatter_money_scores.png`, `outputs/fig_quadrants_money_imdb.png`.

## Text features (Tag Genome + VADER)

**Measures.** Per movie (up to 40 reviews): mean token length; type–token ratio on **fixed 100-token samples**; mean VADER compound. Aggregated by genre.

| Check | N | ρ | p_null |
|---|---:|---:|---:|
| VADER ↔ log(gross/budget) | 2 070 | 0.127 | 0.001 |
| VADER ↔ IMDb | 2 070 | 0.218 | 0.001 |
| Lexical diversity ↔ log(gross/budget) | 2 070 | −0.160 | 0.001 |
| Review length ↔ log(gross/budget) | 2 070 | −0.105 | 0.001 |

Comedy/Romance reviews are more positive on VADER than Thriller/Crime (see `genre_text_features.csv`). Lexical diversity barely varies by genre once length is fixed.

**Validation**

- **Hand labels:** 50 random Tag Genome reviews exported to `build/imdb/features/reviews_private/hand_label_sample_50.csv` (gitignored text). Template without text: `outputs/hand_label_template_50.csv`. **Agreement not yet reported — labels awaiting fill-in.**
- **Automatic transfer check:** VADER vs Stanford Large Movie Review gold polarity on 2 000 docs: accuracy **0.733** on 1 988 non-neutral predictions (0.6% VADER-neutral). Different corpus than Tag Genome.

**What it misses.** VADER is a general lexicon, not movie-specific; Tag Genome has no stars or dates; short reviews drop out of the diversity sample.

## Idea 2 (exploratory) — critic vs audience language

**Measures.** Term rates per 10 000 words in Wikipedia reception text vs Tag Genome reviews for **1 829** shared movies; log–log scatter; person/title tokens removed; **min count = 30 in both** corpora (1 944 terms).

- Critic-heavy terms are encyclopedic/score meta-language: *grossed*, *consensus*, *reception*, *tomatoes*, *ebert*, …
- Audience-heavy terms are experiential: *awesome*, *really*, *watched*, *loved*, *i'm*, …
- Near-diagonal shared terms include *making*, *special*, *genre*, *cinema*, *trailer*.

**What it misses.** Reception sections mix quotes with Wikipedia voice; name filtering is heuristic, not NER. Short windows for inspecting top terms are kept private under `build/` and are not committed.

Figure: `outputs/fig_scattertext_critic_audience.png`.

## How to re-run

Data lives in `build/imdb/` (gitignored). From the repo root:

```bash
python3 scripts/get_data.py --data-dir build/imdb --status
python3 scripts/get_data.py --data-dir build/imdb genome   # if missing
cd build/imdb && ../../.venv/bin/python ../../scripts/imdb/parse_infobox.py
cd ../.. && PYTHONPATH=. .venv/bin/python -m analysis.reviews.run_exploration
```

Then open `01_exploration.ipynb`.

## Limitations and open questions

1. Hand-label the 50-review sample and report agreement / κ with VADER.
2. Does the weak money↔reception link strengthen in the flop-rich 10k tier once Wikipedia money is hand-checked (not only Wikidata)?
3. Franchise / release-month / awards leakage into both money and scores is still open at project level.
4. Critic **sentiment** (not only section length / scores) on Wikipedia reception text is unused beyond the vocabulary scatter.
5. True marketing spend remains unavailable; every “profit” claim is a proxy.

## What is deliberately not in git

Raw Tag Genome / Stanford review text, Wikipedia article wikitext, reception full text, screenplays, and the hand-label CSV **with** review text. Only code, docs, aggregate CSVs and figures are committed.
