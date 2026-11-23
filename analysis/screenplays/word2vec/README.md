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

So "the signal survives the tier control" only means the tier dummy is gone. Popularity isn't: inside 100k+, votes alone separate hits from flops far better than any text representation. Adding the vector to a cross-validated baseline that includes log(votes) doesn't help. For Action 100k+, the baseline with votes gives 0.795, and with the vector added 0.781 (dialogue) and 0.789 (description). Votes are partly the outcome, since a film that more people saw has a larger gross, so this isn't a confounder to adjust away. But it means the AUC can't tell text that predicts success apart from text that marks a widely seen film.

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

The four tests above cover Comedy and Action, with Drama as a check. `all_genres.py` asks the same hit versus flop question in every genre with at least 20 hits and 20 flops, so the screenplay result can sit next to the genre result (Q1). It uses the same labels (top and bottom quarter of ln(gross/budget) within genre x era, D11 to D13) and the same script vectors, but no classifier: only cosine similarity and a label shuffle, which are course methods.

```bash
IMDB_DATA_DIR=/path/to/data/raw python -I analysis/screenplays/word2vec/all_genres.py   # about 1 min, numpy only
```

It reads one space, window 5 and seed 1 (`features/script_w2v/vectors_w5_s1.npz`), not the mean over five seeds. Vectors are centred on the mean of all 2188 scripts first, because without that every pair of scripts has a cosine near 0.9. The statistic is the separation: mean cosine within hits and within flops, minus mean cosine between a hit and a flop. It's above 0 when hit scripts resemble each other, and flop scripts each other, more than they resemble scripts in the other group. The null shuffles the hit and flop labels 2000 times inside each era cell, so the smallest possible p is 0.0005. The last column runs the same statistic for "films of this genre" against the rest of the 1598 films, to show how much genre the vectors carry on the same scale.

| Genre (hits/flops) | Dialogue | p | Description | p | Genre vs rest, dialogue / description |
|---|---|---|---|---|---|
| Drama (223/223) | +0.003 | 0.056 | +0.000 | 0.32 | +0.040 / +0.075 |
| Comedy (136/136) | +0.008 | 0.0080 | +0.004 | 0.058 | +0.064 / +0.074 |
| Action (108/108) | +0.019 | 0.0005 | +0.009 | 0.0085 | +0.095 / +0.127 |
| Crime (85/85) | -0.004 | 0.91 | +0.006 | 0.068 | +0.032 / +0.023 |
| Adventure (84/84) | +0.002 | 0.25 | +0.001 | 0.30 | +0.075 / +0.076 |
| Thriller (74/74) | +0.002 | 0.25 | +0.005 | 0.11 | +0.013 / +0.009 |
| Romance (59/59) | -0.001 | 0.49 | -0.005 | 0.79 | +0.020 / +0.048 |
| Horror (52/52) | +0.049 | 0.0005 | +0.058 | 0.0005 | +0.022 / +0.027 |
| Mystery (50/50) | +0.016 | 0.025 | +0.026 | 0.0050 | +0.011 / +0.010 |
| Sci-Fi (41/41) | -0.006 | 0.82 | +0.007 | 0.15 | +0.053 / +0.028 |
| Biography (40/40) | -0.009 | 0.87 | -0.004 | 0.68 | +0.020 / +0.032 |
| Fantasy (35/35) | +0.016 | 0.093 | +0.018 | 0.090 | +0.035 / +0.018 |

That is 24 tests. After a Holm correction over them, three pass at 0.05: Action dialogue and both Horror layers (`p_holm` in `results/all_genres.csv`). Groups under 50 are below D14's size for a main result and are marked `small`.

The genre signal is much larger than the hit versus flop signal. In every genre but Horror and Mystery, the vectors separate the genre from the rest more than they separate its hits from its flops, in most layers by five times or more (Thriller description and Fantasy are the closest). Every genre-versus-rest test is far beyond its shuffle (z from 6.9 to 176, `results/all_genres_genre.csv`). So the script vector mostly knows what kind of film it is, which is the variable Q1 already measures, and much less whether the film paid off.

Two checks on the four genres with a signal. "Era x tier" shuffles within era x vote tier, so the null keeps the tier mix of hits and flops (D10). "No franchise" drops films with a Wikidata "follows" or "franchise" statement, the flag of `scripts_text/s10_franchise.py` (D26).

| Genre | Layer | p, primary | p, era x tier | Hits/flops without franchise | p, without franchise |
|---|---|---|---|---|---|
| Comedy | dialogue | 0.0080 | 0.0060 | 105/129 | 0.19 |
| Comedy | description | 0.058 | 0.029 | 105/129 | 0.12 |
| Action | dialogue | 0.0005 | 0.11 | 51/94 | 0.16 |
| Action | description | 0.0085 | 0.080 | 51/94 | 0.72 |
| Horror | dialogue | 0.0005 | 0.0010 | 37/45 | 0.019 |
| Horror | description | 0.0005 | 0.0005 | 37/45 | 0.0020 |
| Mystery | dialogue | 0.025 | 0.041 | 45/47 | 0.081 |
| Mystery | description | 0.0050 | 0.029 | 45/47 | 0.0040 |

Action doesn't survive either check. Inside a tier the shuffle already gives a separation of about half the observed one, so most of the Action signal is the vote tier, as the 100k+ section above found for the classifier. Comedy survives the tier check but not the franchise check. Horror survives both, and it's the only genre where hits and flops separate more than Horror separates from other genres. The most similar pairs among the 104 Horror scripts are sequels of one another (Insidious and its Chapter 2, two Scream films) and creature films that are all flops (Alien, Deep Rising, Below, The Thing). Reading the title lists, the hits are mostly slasher and haunted-house films and the flops mostly creature and sci-fi horror. That's a look at the titles, not a measurement, but it matches the Horror subgenre and Blumhouse results of Q1: the script picks up the kind of Horror film, and the cheap kind paid off.

This measures whether hit scripts resemble each other more than they resemble flops, in a space trained on all 2188 scripts. It misses anything a mean vector loses: structure, plot and order. It can't tell a script that causes success from one that marks a subgenre, a budget level or a franchise, and the checks above show those carry most of what it finds. It uses one seed, where the primary tests average five.
