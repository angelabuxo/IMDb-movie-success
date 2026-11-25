# Word2Vec script vectors: the four tests

This folder runs the test set up in `../DECISIONS.md`. The settings it left open were fixed and committed before any AUC was computed (section "Test settings fixed before any AUC"). Every number below comes from `results/*.csv`.

## How to run it

gensim 4.4.0 doesn't build on the repo's Python 3.15, so these scripts run in a separate Python 3.13 environment:

```bash
uv venv -p python3.13 .venv-w2v && uv pip install -p .venv-w2v/bin/python gensim==4.4.0 scikit-learn==1.9.1 pandas numpy
export IMDB_DATA_DIR=/path/to/data/raw
.venv-w2v/bin/python -I analysis/screenplays/word2vec/corpus.py      # 22 s, writes private/corpus.txt and counts.pkl
.venv-w2v/bin/python -I analysis/screenplays/word2vec/train.py --window 5 --seed 1   # about 12 min per space, workers=1
.venv-w2v/bin/python -I analysis/screenplays/word2vec/vectors.py     # every trained space in private/
.venv-w2v/bin/python -I analysis/screenplays/word2vec/run_tests.py   # about 1.5 h on 9 cores
.venv-w2v/bin/python -I analysis/screenplays/word2vec/subset_tests.py  # 100k+ shuffle and vector+baseline interval, about 50 min on 9 cores
.venv-w2v/bin/python -I analysis/screenplays/word2vec/votes_check.py   # about 1 min
```

The primary run uses windows 5, 2 and 10, each with seeds 1 to 5, so 15 spaces in all. A rerun of the same seed gives identical vectors, which was checked on a 100,000-line slice. The corpus has 19,484,090 dialogue tokens and 24,756,749 description tokens, the same counts as `preprocess/preprocess_summary.csv`. Everything in `private/` holds script text or is derived from it and is gitignored.

## Results

AUC of hit versus flop in repeated 5-fold cross-validation. For Word2Vec the AUC is the mean over 5 seeds. "Shuffle p" comes from 200 label permutations, so 0.005 is the smallest possible value.

| Genre | Layer | Word2Vec | shuffle p | TF-IDF | shuffle p | Baseline (tier + genre + era) |
|---|---|---|---|---|---|---|
| Comedy (136/136) | dialogue | 0.613 | 0.005 | 0.539 | 0.18 | 0.782 |
| Comedy | description | 0.584 | 0.005 | 0.505 | 0.46 | 0.782 |
| Action (108/108) | dialogue | 0.640 | 0.005 | 0.618 | 0.01 | 0.706 |
| Action | description | 0.664 | 0.005 | 0.615 | 0.01 | 0.706 |
| Drama (223/223, exploratory) | dialogue | 0.563 | 0.01 | 0.575 | 0.01 | 0.786 |
| Drama | description | 0.559 | 0.045 | 0.528 | 0.15 | 0.786 |

In all four primary tests the Word2Vec vector beats the shuffle. It never beats the baseline. Adding the vector to the baseline features gives 0.764 and 0.766 for Comedy, below the baseline alone, and 0.769 and 0.785 for Action, up from 0.706.

Most of the baseline is the vote tier. Of the 136 Comedy hits, 119 are 100k+ films, against 42 of the 136 flops, and Action has 104 against 45. Within 100k+ films only, the baseline drops to about 0.52. There the Word2Vec vector still gives 0.59 and 0.57 for Comedy and 0.63 and 0.66 for Action (dialogue, description). The shuffle test for that subset is below.

## Inside the 100k+ tier, with a shuffle

`subset_tests.py` reruns the primary pipeline on the 100k+ films only, with 200 label permutations. The baseline there is genre + era, since the tier column is constant. The folds are cut from the smaller set with the same settings, so they aren't literally the folds of `primary.csv`. The observed AUCs match `robustness.csv`.

| Genre | Layer | Word2Vec | shuffle p | TF-IDF | shuffle p | Baseline (genre + era) |
|---|---|---|---|---|---|---|
| Comedy (119/42) | dialogue | 0.588 | 0.030 | 0.506 | 0.45 | 0.519 |
| Comedy | description | 0.567 | 0.075 | 0.495 | 0.51 | 0.519 |
| Action (104/45) | dialogue | 0.634 | 0.010 | 0.510 | 0.37 | 0.518 |
| Action | description | 0.660 | 0.005 | 0.586 | 0.030 | 0.518 |

The baseline itself is no better than a shuffle there (p 0.43 and 0.40). The Word2Vec vector beats the shuffle in three of the four tests. Comedy description, at p = 0.075, doesn't. The signal is clearest in Action, where TF-IDF also survives in the description layer.

## How much popularity is left inside the tier

The tier is a split at 100,000 votes, and votes still vary a lot above it. `votes_check.py` scores hits against flops by log(votes) alone, with no model:

| Genre | Set | Median votes, hits | Median votes, flops | AUC of log(votes) |
|---|---|---|---|---|
| Comedy | all labelled | 310,393 | 65,747 | 0.882 |
| Comedy | 100k+ | 355,030 | 145,148 | 0.818 |
| Action | all labelled | 522,416 | 83,034 | 0.926 |
| Action | 100k+ | 535,638 | 159,953 | 0.877 |

So "the signal survives the tier control" only means the tier dummy is gone. Popularity isn't: inside 100k+, votes alone separate hits from flops far better than any text representation. We also added the vector to a cross-validated baseline that includes log(votes) (`results/votes_check.csv`):

| Genre | Set | Baseline + votes | With dialogue vector | With description vector |
|---|---|---|---|---|
| Comedy | all labelled | 0.815 | 0.841 | 0.839 |
| Comedy | 100k+ | 0.743 | 0.666 | 0.730 |
| Action | all labelled | 0.879 | 0.864 | 0.872 |
| Action | 100k+ | 0.795 | 0.781 | 0.789 |

The AUC drops in three of the four rows. In Comedy over all labelled films it rises by about 0.025, but 0.841 is still below the 0.882 of log(votes) alone. The baseline with votes is below log(votes) alone in every row, so the classifier doesn't use the votes fully, and part of that rise could come from this. These are observed AUCs with no shuffle or interval, so neither the rise nor the drops are tested.

Votes are partly the outcome, since a film that more people saw has a larger gross, so this isn't a confounder to adjust away. But it means the AUC can't tell text that predicts success apart from text that marks a widely seen film.

## Vector + baseline, with an interval

`subset_tests.py` also bootstraps films (2000 resamples) over the out-of-fold scores of `w2v+baseline` (`results/combo_interval.csv`):

| Genre | Layer | Vector + baseline | 95 % interval | Gain over baseline | 95 % interval |
|---|---|---|---|---|---|
| Comedy | dialogue | 0.764 | 0.712 to 0.813 | -0.017 | -0.057 to +0.021 |
| Comedy | description | 0.766 | 0.714 to 0.815 | -0.015 | -0.050 to +0.018 |
| Action | dialogue | 0.769 | 0.715 to 0.820 | +0.063 | +0.013 to +0.109 |
| Action | description | 0.785 | 0.732 to 0.835 | +0.079 | +0.035 to +0.125 |

In Action the vector adds to the tier + genre + era baseline, and the interval excludes zero. In Comedy it doesn't add anything. The bootstrap doesn't refit, so it ignores training variance and is too narrow if anything.

## The hypothesis

The prediction was that dialogue separates Comedy hits from flops better than description does, and the reverse for Action. Word2Vec points that way in both genres, but neither difference is distinguishable from zero:

| Genre | Representation | AUC(dialogue) - AUC(description) | 95 % bootstrap interval |
|---|---|---|---|
| Comedy | Word2Vec | +0.030 | -0.026 to +0.085 |
| Comedy | TF-IDF | +0.034 | -0.021 to +0.089 |
| Action | Word2Vec | -0.024 | -0.084 to +0.040 |
| Action | TF-IDF | +0.003 | -0.057 to +0.060 |

The Comedy direction also flips under the plain mean, where description gives 0.631 and dialogue 0.585. Windows 2 and 10 keep the window-5 direction in both primary genres (`results/robustness.csv`). So the data don't support the layer hypothesis. With 108 to 136 films per group, the test can't tell differences of this size apart.

## What this measures and what it misses

The AUC says the vector holds something that differs between hits and flops of the same genre and era. It doesn't say what. The vector could be picking up anything correlated with a film's popularity, which the tier result suggests. The test films also took part in the unsupervised training, and gross over budget isn't profit. The bootstrap interval ignores training variance, so it is too narrow if anything.

## Every genre, with cosine similarity and a shuffle (post hoc, exploratory)

The four tests above cover Comedy and Action, with Drama as a check. `all_genres.py` asks the same hit versus flop question in every genre with at least 50 hits and 50 flops, the group's size rule for a result (D14), so the screenplay result can sit next to the genre result (Q1). It uses the same labels (top and bottom quarter of ln(gross/budget) within genre x era, D11 to D13) and the same script vectors, but no classifier: only cosine similarity and a label shuffle, which are course methods.

```bash
IMDB_DATA_DIR=/path/to/data/raw python -I analysis/screenplays/word2vec/all_genres.py   # about 1 min per seed, numpy only
```

It reads every window-5 space in `features/script_w2v/` (`vectors_w5_s1.npz` to `vectors_w5_s5.npz`, the five seeds of the primary tests, or another folder set with `W2V_VECTOR_DIR`) and averages the cosine over seeds. The separation is linear in the cosine, so that is the mean of the five per-seed separations, with every seed seeing the same label shuffles. Vectors are centred on the mean of all 2188 scripts first, because without that every pair of scripts has a cosine near 0.9. The statistic is the separation: mean cosine within hits and within flops, minus mean cosine between a hit and a flop. It's above 0 when hit scripts resemble each other, and flop scripts each other, more than they resemble scripts in the other group. The null shuffles the hit and flop labels 2000 times inside each era cell, so the smallest possible p is 0.0005. The last column runs the same statistic for "films of this genre" against the rest of the 1598 films, to show how much genre the vectors carry on the same scale.

| Genre (hits/flops) | Dialogue | p | Description | p | Genre vs rest, dialogue / description |
|---|---|---|---|---|---|
| Drama (223/223) | +0.003 | 0.049 | +0.001 | 0.29 | +0.040 / +0.075 |
| Comedy (136/136) | +0.008 | 0.0065 | +0.004 | 0.061 | +0.064 / +0.074 |
| Action (108/108) | +0.018 | 0.0010 | +0.010 | 0.0085 | +0.095 / +0.127 |
| Crime (85/85) | -0.005 | 0.91 | +0.006 | 0.076 | +0.032 / +0.023 |
| Adventure (84/84) | +0.002 | 0.24 | +0.001 | 0.30 | +0.074 / +0.076 |
| Thriller (74/74) | +0.002 | 0.24 | +0.005 | 0.11 | +0.013 / +0.009 |
| Romance (59/59) | -0.001 | 0.44 | -0.005 | 0.83 | +0.020 / +0.048 |
| Horror (52/52) | +0.049 | 0.0005 | +0.059 | 0.0005 | +0.022 / +0.028 |
| Mystery (50/50) | +0.016 | 0.025 | +0.025 | 0.0060 | +0.012 / +0.010 |
| Sci-Fi (41/41), below D14 | -0.006 | 0.84 | +0.006 | 0.16 | +0.052 / +0.028 |
| Biography (40/40), below D14 | -0.009 | 0.85 | -0.004 | 0.68 | +0.020 / +0.032 |
| Fantasy (35/35), below D14 | +0.015 | 0.10 | +0.017 | 0.095 | +0.035 / +0.018 |

The numbers are averaged over the five seeds. A run on seed 1 alone gave the same picture: no separation moved by more than 0.0012, and the same three tests pass the correction. Nine genres meet D14, which makes 18 tests. After a Holm correction over them, three pass at 0.05: Action dialogue and both Horror layers (`p_holm` in `results/all_genres.csv`). Sci-Fi, Biography and Fantasy have 35 to 41 films per group. They're computed and shown, marked `small`, but they aren't tests and aren't in the correction. Their shuffle null is about 3 to 5 times wider than Comedy's, so a Comedy-sized difference would be invisible there and their null results can't be read either way. The script also computes nothing under 20 per group, which is a floor for showing a row and not a group decision.

The genre signal is much larger than the hit versus flop signal. In every genre but Horror and Mystery, the vectors separate the genre from the rest more than they separate its hits from its flops, in most layers by five times or more (Thriller description and Fantasy are the closest). Every genre-versus-rest test is far beyond its shuffle (z from 7.1 to 176, `results/all_genres_genre.csv`). So the script vector mostly knows what kind of film it is, which is the variable Q1 already measures, and much less whether the film paid off.

Two checks on the four genres with a signal. "Era x tier" shuffles within era x vote tier, so the null keeps the tier mix of hits and flops (D10). "No franchise" drops films with a Wikidata "follows" or "franchise" statement, the flag of `scripts_text/s10_franchise.py` (D26).

| Genre | Layer | p, primary | p, era x tier | Hits/flops without franchise | p, without franchise |
|---|---|---|---|---|---|
| Comedy | dialogue | 0.0065 | 0.0050 | 105/129 | 0.17 |
| Comedy | description | 0.061 | 0.031 | 105/129 | 0.12 |
| Action | dialogue | 0.0010 | 0.11 | 51/94 | 0.16 |
| Action | description | 0.0085 | 0.079 | 51/94 | 0.72 |
| Horror | dialogue | 0.0005 | 0.0010 | 37/45 | 0.021 |
| Horror | description | 0.0005 | 0.0005 | 37/45 | 0.0020 |
| Mystery | dialogue | 0.025 | 0.038 | 45/47 | 0.077 |
| Mystery | description | 0.0060 | 0.033 | 45/47 | 0.0040 |

Action doesn't survive either check. Inside a tier the shuffle already gives a separation of about half the observed one, so most of the Action signal is the vote tier, as the 100k+ section above found for the classifier. Comedy survives the tier check but not the franchise check. Horror survives the tier check, and it's the only genre where hits and flops separate more than Horror separates from other genres. Without franchise films its description layer still passes the Holm correction (0.036), but its dialogue layer doesn't (p 0.021, Holm 0.34), so about half of the dialogue separation goes with the sequels.

`horror_check.py` asks what the Horror split is made of, with the TMDB keyword subgenres of `analysis/genre/q1_tmdb.py`, whose keyword lists were fixed there before looking. The hits are more often supernatural (37 % of the 52 hits against 15 % of the 52 flops) and slasher (38 % against 12 %), and less often creature films (27 % against 40 %). The median budget is $10.4M for hits and $15.0M for flops, which partly follows from defining hits on gross over budget. But none of these explains the separation. Shuffling within era x each subgenre flag, or within era x the Horror budget median, still leaves it far above the null (z from 4.1 to 10.2, every p at most 0.0035, `results/horror_check.csv`). The largest null comes from the slasher flag, at a sixth of the observed separation in dialogue. So Horror hit scripts resemble each other in a way the subgenre labels and the budget don't capture, and this test can't say what it is. Stratifying on all flags at once would leave mostly single-film strata, where the shuffle can't move anything, so it isn't done.

This measures whether hit scripts resemble each other more than they resemble flops, in a space trained on all 2188 scripts. It misses anything a mean vector loses: structure, plot and order. It can't tell a script that causes success from one that marks a subgenre, a budget level or a franchise, and the checks above show those carry most of what it finds.

## The same test, with budget held fixed (post hoc, D29)

`budget_matched.py` repeats the test above but shuffles labels only within genre x era x budget third. Its criteria (D29) were written after a one-seed run and before this five-seed run. The era-only run reproduces the separations of `all_genres.csv`.

| Genre (hits/flops) | Layer | Separation | 95 % of shuffles | p | Holm p (18 tests) |
|---|---|---|---|---|---|
| Action (108/108) | dialogue | +0.0183 | 0.0221 | 0.0960 | 0.960 |
| Action (108/108) | description | +0.0097 | 0.0152 | 0.2224 | 1.000 |
| Adventure (84/84) | dialogue | +0.0022 | 0.0091 | 0.2649 | 1.000 |
| Adventure (84/84) | description | +0.0008 | 0.0069 | 0.3483 | 1.000 |
| Comedy (136/136) | dialogue | +0.0083 | 0.0046 | 0.0075 | 0.120 |
| Comedy (136/136) | description | +0.0041 | 0.0047 | 0.0670 | 0.806 |
| Crime (85/85) | dialogue | -0.0045 | 0.0064 | 0.9065 | 1.000 |
| Crime (85/85) | description | +0.0058 | 0.0071 | 0.0850 | 0.935 |
| Drama (223/223) | dialogue | +0.0034 | 0.0038 | 0.0620 | 0.806 |
| Drama (223/223) | description | +0.0005 | 0.0031 | 0.2914 | 1.000 |
| Horror (52/52) | dialogue | +0.0485 | 0.0239 | 0.0010 | 0.017 |
| Horror (52/52) | description | +0.0586 | 0.0246 | 0.0005 | 0.009 |
| Mystery (50/50) | dialogue | +0.0159 | 0.0154 | 0.0470 | 0.658 |
| Mystery (50/50) | description | +0.0245 | 0.0134 | 0.0105 | 0.157 |
| Romance (59/59) | dialogue | -0.0005 | 0.0110 | 0.4228 | 1.000 |
| Romance (59/59) | description | -0.0050 | 0.0104 | 0.8486 | 1.000 |
| Thriller (74/74) | dialogue | +0.0023 | 0.0084 | 0.2009 | 1.000 |
| Thriller (74/74) | description | +0.0050 | 0.0085 | 0.1209 | 1.000 |

Only Horror passes. Its description layer also passes with budget halves and quarters, without franchise films (p 0.0045), in the 100k+ films (p 0.041, 46/20) and with the vote tier kept in the shuffle (p 0.0075, 58 % of films movable). Its dialogue layer misses with budget quarters (Holm 0.051) and in the 100k+ films (p 0.077), so it isn't reported as a finding. All five seeds agree. Of 6,006 description words only *closet* beats the family-wise shuffle threshold, and no dialogue word does, so the test mostly can't say what differs. A failed test isn't evidence that scripts don't matter.

`prerelease.py` has the two other checks. Against era, other genres and log budget (linear), the vectors raise Action's AUC by 0.084 (dialogue) and 0.088 (description) with intervals above zero, which disagrees with the shuffle above, so no Action claim is made. And films with a script are the popular ones: 36 % of hits have one against 13 % of flops, 49 % against 42 % among 100k+ films.
