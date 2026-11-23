# Figures for the screenplay question (Q3)

Three candidate figures. They are drawn by `make_figures.py` from the committed tables in `../word2vec/results/` only; nothing is recomputed. PNG for a quick look, PDF (vector) for the paper.

```bash
.venv/bin/python scripts-analysis/figures/make_figures.py   # from the repo root
```

The question the figures answer: within one genre and era, does the vector of a screenplay tell a hit from a flop, and does it do so differently for dialogue and for scene description?

All AUC values are the area under the ROC curve for hit versus flop in repeated 5-fold cross-validation (0.5 is chance, 1.0 is perfect). Hits and flops are the top and bottom quarter of ln(gross/budget) within genre and era. Comedy has 136 hits and 136 flops, Action 108 and 108.

## Which one to use: Figure 1

The course asks for a figure that shows an effect against its null, with the check next to the claim, and for one story told clearly. Figure 1 does that in one image. It shows that the script vector beats chance, that it does not beat simple metadata, and that popularity carries more than the text. Figures 2 and 3 are supporting: use Figure 2 as a sentence or a small panel, and Figure 3 only if the discussion needs the point about popularity.

If the group's paper compares the three questions (genre, people, script) on the same films, Figure 1 should become one row of that shared comparison, with the same null and the same film set. That needs the other two questions' numbers, which are not in this folder.

## Figure 1: `fig1_signal_vs_null` (recommended)

**What it shows.** For each of the four tests (Comedy and Action, each in the dialogue and the description layer), the AUC of several predictors next to the AUC you get when the hit/flop labels are shuffled.

**Axes.** x: AUC of hit versus flop (0.5 is chance). y: the four tests. Panel A uses all labelled films. Panel B uses only films with 100,000 or more IMDb votes (Comedy 119 hits and 42 flops, Action 104 and 45).

**Marks.**
- Blue dot: Word2Vec script vector (TF-IDF-weighted mean of word vectors, window 5, mean over 5 seeds).
- Orange open square: TF-IDF representation of the same words, no Word2Vec.
- Grey band: 2.5th to 97.5th percentile of 200 label shuffles of the whole pipeline. A dot to the right of the band beats chance.
- Dark grey diamond: metadata baseline. In A it is vote tier + era + the film's other genres. In B the tier is constant, so it is era + other genres.
- Black cross: log(votes) alone, with no model.
- Right of each panel: the shuffle p of the Word2Vec dot. With 200 shuffles the smallest possible value is 0.005.

**What to read.** The Word2Vec vector beats the shuffle (p below 0.05) in 7 of the 8 cases; the exception is Comedy description in B, p = 0.075. So the script vector carries a weak signal beyond chance. They sit left of the diamond and the cross in A: simple metadata and, above all, the number of votes separate hits from flops better than the text does. In B the metadata baseline falls to chance but the votes alone still reach 0.82 and 0.88, so popularity is still there inside the tier. The signal is clearest in Action.

**What it misses.** It does not say what the vector picks up, and the vector may mark "a widely seen film" and not "a good script". Votes are partly a consequence of success, so they are not a confounder to adjust away. The 200 shuffles give p resolution only down to 0.005. The AUC of log(votes) is a raw score with no fitted model, so it is not strictly comparable with the cross-validated ones.

## Figure 2: `fig2_layer_hypothesis`

**What it shows.** The hypothesis written before any result: dialogue separates Comedy hits from flops better than description does, and the reverse for Action.

**Axes.** x: AUC(dialogue) minus AUC(description), with a 95 % paired bootstrap interval over films (2000 resamples). Right of 0 means dialogue is better. Rows: Comedy, Action and Drama (exploratory, no prediction), each for Word2Vec (blue dot) and TF-IDF (orange square). The number at the right is the estimate.

**What to read.** The Word2Vec estimates go in the predicted direction (+0.030 in Comedy, -0.024 in Action), but every interval includes zero, and Action with TF-IDF is +0.003, on the other side. The hypothesis is not supported, and with 108 to 136 films per group the test cannot see differences of this size. A negative result with a check is a fine result. Under the plain mean instead of the TF-IDF weighting the Comedy direction flips (description 0.631, dialogue 0.585; `../word2vec/results/robustness.csv`).

**What it misses.** The interval ignores the variance from training Word2Vec, so it is too narrow if anything. Drama TF-IDF reaches the edge of zero (+0.001 to +0.095): it is exploratory and one of many comparisons, so it is not a finding.

## Figure 3: `fig3_added_value`

**What it shows.** Whether adding the script vector to the metadata helps, and how that depends on popularity.

**Axes.** x: change in AUC when the Word2Vec vector is added. Rows: the four tests. Blue dot with a bar: added to vote tier + era + other genres, with a 95 % bootstrap interval. Orange open square: added to the same features plus log(votes), a point estimate only (all labelled films, no interval).

**What to read.** In Action the vector adds +0.063 (dialogue) and +0.080 (description) to the tier baseline, with intervals that exclude zero. Once log(votes) is in the baseline the gain is gone (-0.016, -0.007). In Comedy the vector does not help the tier baseline (-0.017, -0.015) and gains about +0.025 when votes are in; this has no interval and we have not tested it, so we do not interpret it. So the script adds something over a coarse popularity marker (the tier) in Action, but not over the votes.

**What it misses.** The orange points have no uncertainty. The baseline with votes (0.815 in Comedy) does worse than log(votes) alone (0.882), so the extra features add noise to that reference.

## What was not drawn

The hit and flop word lists (`../words/`) are exploratory: the calibrated ranking was chosen after looking at the uncalibrated lists, and the reading of the context lines covers 18 words by one reader. They fit in one sentence of the text, not in a figure.

## Colour and accessibility

Categorical slots 1 and 2 of the validated default palette (blue `#2a78d6`, orange `#eb6834`) mark which representation. References are grey or black. Markers also differ in shape, so the figures work in black and white. Light surface only; there is no dark version.

## Provenance

Sources: `primary.csv`, `null_*.csv`, `subset_100k.csv`, `null_100k_*.csv`, `votes_check.csv`, `combo_interval.csv`, `contrast.csv`. These come from the Word2Vec pipeline in `../word2vec/`. The pipeline's AUCs were not independently reproduced when these figures were drawn.
