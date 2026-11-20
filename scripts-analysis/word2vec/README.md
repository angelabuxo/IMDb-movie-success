# Word2Vec script vectors: the four tests

This folder runs the test Angi set up in `../DECISIONS.md`. The settings it left open were fixed and committed before any AUC was computed (section "Test settings fixed before any AUC"). Every number below comes from `results/*.csv`.

## How to run it

gensim 4.4.0 doesn't build on the repo's Python 3.15, so these scripts run in a separate Python 3.13 environment:

```bash
uv venv -p python3.13 .venv-w2v && uv pip install -p .venv-w2v/bin/python gensim==4.4.0 scikit-learn==1.9.1 pandas numpy
export IMDB_DATA_DIR=/path/to/data/raw
.venv-w2v/bin/python -I scripts-analysis/word2vec/corpus.py      # 22 s, writes private/corpus.txt and counts.pkl
.venv-w2v/bin/python -I scripts-analysis/word2vec/train.py --window 5 --seed 1   # about 12 min per space, workers=1
.venv-w2v/bin/python -I scripts-analysis/word2vec/vectors.py     # every trained space in private/
.venv-w2v/bin/python -I scripts-analysis/word2vec/run_tests.py   # about 1.5 h on 9 cores
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

Most of the baseline is the vote tier. Of the 136 Comedy hits, 119 are 100k+ films, against 42 of the 136 flops, and Action has 104 against 45. Within 100k+ films only, the baseline drops to about 0.52. There the Word2Vec vector still gives 0.59 and 0.57 for Comedy and 0.63 and 0.66 for Action (dialogue, description), though without a shuffle test in that subset.

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
