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

## Open decisions (Angi decides)

1. **Genre(s)** and the rule for choosing them, written before seeing results.
2. **Definition of hit and flop:** gross >= 2x budget (absolute) vs top/bottom quarter of log(gross/budget) within genre (relative). Note: gross is not profit.
3. **The middle:** drop it, or keep it as a third group.
4. **Franchises and era:** not the focus, but report results with and without them as a cheap robustness check.
5. **Primary layer:** dialogue, description or both.
6. **Aggregation:** plain mean vs TF-IDF-weighted mean (and stopword handling).
7. **Window size:** narrow (style/syntax) vs wide (topic).
8. **Training data:** our scripts vs a pretrained space (GloVe is trained on Wikipedia and news, not screenplays).
9. **Preprocessing:** lowercase, character names, stopwords, lemmatisation.
10. **Hypothesis**, e.g. "description separates hits from flops more than dialogue does".

## Questions the vectors cannot answer

- Which words drive a difference. That is Scattertext's job.
- Profit. The vector alone says nothing about it.

## Rules for this folder

- Screenplays are copyrighted (MovieSum is CC BY-NC 4.0). No script text is committed. Anything containing script text goes in `scripts-analysis/private/` (gitignored).
- Snippets shown in chat are at most 15 words.
