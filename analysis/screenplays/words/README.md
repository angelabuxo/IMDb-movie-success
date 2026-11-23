# Which words separate hits from flops

This was requested on 2026-11-21 as the follow-up to the Word2Vec tests: a Scattertext-style look at Action and Comedy, on the same hit/flop labels, the 1598-film set and the same preprocessing (T = 20 names filter), with a shuffle check and a look back at the text. Every number below comes from `results/*.csv`, written by `words.py`.

## How it works

Scattertext 0.2.2 pins scipy 1.13, which doesn't build on Python 3.13, so `words.py` computes the statistic itself. It's the one Scattertext calls `LogOddsRatioInformativeDirichletPrior`: the log-odds ratio with an informative Dirichlet prior from Monroe, Colaresi and Quinn (2008), reported as a z-score. The prior is each word's count over the 1598 films of the final set, per layer. The two groups are the labelled hits and flops of one genre (Action 108/108, Comedy 136/136), and the counts come from `private/counts.pkl`, so the preprocessing is the one the vectors use.

That z counts every token as an independent draw, and tokens aren't independent: one film that repeats a word a thousand times moves it a long way. So the test is a film-level shuffle. Labels are permuted 1000 times within genre x era cells, the cells the labels were cut in, and z is recomputed each time. A word that a few films use heavily gets a wide shuffle distribution, so the lists are ranked by the calibrated z, which is the observed z minus that word's shuffle mean, divided by its shuffle sd. Words in fewer than 10 of the genre's labelled films are left out.

```bash
export IMDB_DATA_DIR=/path/to/data/raw
.venv-w2v/bin/python -I scripts-analysis/words/words.py --perms 1000 --kwic   # about 40 s
```

`--kwic` writes `private/kwic/<genre>_<layer>.txt`. For every listed word, that file holds one line of at most 15 words from each of 8 films, 4 hits and 4 flops, each word drawing its own random set of films. It contains script text, it's gitignored, and it must never be committed or pasted anywhere.

## The uncalibrated ranking is mostly artefacts

Ranked by the plain Monroe z, the hit side of all four lists starts with single letters (t, e, h, l, n, r). In Action dialogue, 1334 of the 1802 hit-side uses of "t" come from one film. Those films have letter-spaced or garbled text, the kind the manual check flagged in two scripts. The flop side is mostly first names (peel, kim, nick, steve, burt, larry, roy). In each of the four Monroe top-40 lists, between 33 and 40 words are a character-label token in 20 or more films, so the T = 20 filter keeps them as ordinary words. Those are the same two failure modes the data check describes, and they're why the lists below use the calibrated ranking. The calibrated lists have no single letters, and between 3 and 11 words of each top 40 are frequent label tokens. That choice was made after seeing the Monroe lists, so it isn't a pre-registered setting.

## Does any word beat the shuffle?

No single word does once you account for testing thousands of them. The largest |calibrated z| over all words never beats the shuffle's 95th percentile. Its shuffle p is 0.15 for Action dialogue, 0.10 for Action description, 0.22 for Comedy dialogue and 0.32 for Comedy description.

Whole lists do better. Here is the shuffle p of the k-th largest calibrated z, against the k-th largest under the shuffle:

| Genre | Layer | Side | k = 1 | k = 5 | k = 10 | k = 20 |
|---|---|---|---|---|---|---|
| Action | dialogue | hit | 0.081 | 0.30 | 0.069 | 0.25 |
| Action | dialogue | flop | 0.22 | 0.001 | 0.001 | 0.001 |
| Action | description | hit | 0.049 | 0.014 | 0.037 | 0.024 |
| Action | description | flop | 0.25 | 0.011 | 0.002 | 0.002 |
| Comedy | dialogue | hit | 0.93 | 0.82 | 0.66 | 0.58 |
| Comedy | dialogue | flop | 0.12 | 0.004 | 0.008 | 0.044 |
| Comedy | description | hit | 0.19 | 0.003 | 0.002 | 0.004 |
| Comedy | description | flop | 0.72 | 0.22 | 0.17 | 0.082 |

So in five of the eight lists the top 5 to 20 words separate more than chance would, while no single word stands out. That fits the Word2Vec result: a weak signal spread over many words. The table holds 32 correlated tests, so treat a lone 0.04 with care. The Comedy dialogue hit side, at 0.58 to 0.93, is no different from a shuffle.

## The top words, and what they are in context

The top 10 by calibrated z (full top 20 with counts in `results/top_<genre>_<layer>.csv`):

| Genre | Layer | Hits | Flops |
|---|---|---|---|
| Action | dialogue | perfectly, congratulations, exist, destruction, absolutely, northern, misery, cement, ow, venom | they'd, shoot, where's, crime, wasn't, nobody's, tonight, began, guess, christ |
| Action | description | crashes, oncoming, chases, run, blasts, crash, equipment, traffic, drown, speed | moment, blood, uniforms, printed, silence, couple, lips, takes, attractive, cheeks |
| Comedy | dialogue | texted, day, whoa, curb, bye, embrace, em, naughty, auntie, whining | dollars, understand, die, trap, ones, bleeding, multiple, moral, in, excessive |
| Comedy | description | hops, reacts, frustrated, proud, romantic, concerned, chase, procession, chute, spots | rapid, pay, dental, sound, looming, apparently, low, source, sweat, films |

I read the context lines for 18 of these words. What I found:

- **Action description is the clearest.** Hit scripts describe vehicle and chase set pieces: oncoming trains and cars, crashes, blasts, traffic, speed. Flop scripts lean toward bodies and violence close up (blood, lips, cheeks) and toward introducing characters by their looks ("attractive" is almost always a woman being introduced). "Uniforms" is police and soldiers on both sides. This reads as large-scale spectacle against smaller-scale crime and thriller writing, which could just as well be budget as anything that makes a hit.
- **Action dialogue flops** use crime-story talk (crime boss, crime scene, shoot, bucks, christ, jesus) and contracted speech (they'd, wasn't, where's, gotta). The hit-side words (perfectly, exist, congratulations) are ordinary words in ordinary uses, and that side doesn't beat the shuffle.
- **Comedy description hits** use physical-comedy and reaction directions (hops in, reacts, frustrated, proud). The flop side partly comes from script format, not story. "Films" and "draft" show up in title-page text (production company, draft date) in some flop scripts, and "dental" is in 12 flop films and no hit, mostly as a dentist scene. That side doesn't beat the shuffle at k = 10.
- **Comedy dialogue hits** include "texted", which is in 12 hit films against 1 flop and is a recent word. Era is only controlled in decade bins, so a word like this can carry the newest films inside a bin. The hit side as a whole is no different from a shuffle.

Two checks against the obvious explanations. For the 100k+ tier only, the Monroe z of each listed word keeps the same sign in 159 of the 160 calibrated top-20 words. So the lists aren't just the tier, though inside the tier hits still have far more votes (see `../word2vec/README.md`). The number of hit and flop films containing each word is in the CSVs. Between 2 and 13 of each top 40 sit in fewer than 20 films, and those are the ones to discount.

## What this measures and what it misses

The lists say which words hit and flop scripts of the same genre and era use at different rates. They don't say the words cause anything. Most of them track sub-genre (chase-heavy action against crime thrillers, romantic comedy against other comedy) or budget, and the vector tests already showed that the baseline, which includes the other genres, beats the text. The KWIC reading covers 18 words, 8 lines each, by one reader. The calibrated ranking was chosen after seeing the uncalibrated lists.
