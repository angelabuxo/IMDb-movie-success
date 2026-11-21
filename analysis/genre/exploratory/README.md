# Q1, exploratory: weak evidence, not used for claims

Everything in this folder was run after the main results in [`../README.md`](../README.md) and is kept for the record, not for claims. Each part had a weakness serious enough that we don't build on it:

| Part | Why it is here |
|---|---|
| 1. Genre network and Louvain families | The backbone is a fine description, but the families are no stronger than a rewired network (p 0.62), so testing their returns tests groups that may not exist. |
| 2. Genre assortativity on the people network | The null shuffles away all structure, so a huge z was guaranteed. People repeating genres is expected and is driven by sequels and franchises. A description, not evidence. |
| 3. Tag Genome and plot genre shares, plot clusters | Written after release and carrying reception. Plot clusters are a null result. |
| 4. Horror plot words | Post-release text, a word set picked after seeing the list, and it doesn't replicate in the screenplays (part 7). |
| 5. Genre shares through the follow-ups | Same measure as part 3, with many variants tried. |
| 6. Fixing the shares | Careful, but it repairs a measure the analysis doesn't rely on. The script-predicted shares learn from a target that carries reception. |
| 7. Horror words in the screenplays | The failed replication of part 4. |

The one line worth keeping from here: weighting films by Tag Genome genre shares gives the same genre ranking as IMDb's tags. The matched-pairs check (`../q1_checks.py`) is also exploratory: it's a different estimand, uncorrected for 18 genres, and repeats what the shuffle and regression show.

The notebook `notebooks/q1_genre.ipynb` shows all of this in its exploratory part, after the core sections.

```bash
IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_network          # 4.5 min on the HPC
IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_shares           # 1 min
IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_words            # 5 s once the plot counts are cached
IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_share_followups  # 2.5 min on the HPC
IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_share_fixes      # 1 min
IMDB_DATA_DIR=/path/to/data/raw .venv/bin/python -m analysis.genre.exploratory.q1_horror_scripts   # 30 s
```

`q1_words.py` reads the Wikipedia plots once and caches word counts in `private/` here, which git ignores. No plot text leaves `data/raw`.


`q1_words.py` reads the Wikipedia plots once and caches word counts in `analysis/genre/private/`, which git ignores. No plot text leaves `data/raw`.

## 1. The genre network (weeks 1, 2, 4)

This is the film-genre network projected onto genres. Two genres are linked by the number of films that carry both. Big genres share many films just because they're big, so every pair is compared with a curveball null. The null swaps genres between films while keeping each film's number of genres and each genre's number of films, over 1,000 samples.

- **Backbone.** 41 of the 153 genre pairs co-occur more than the null predicts (Holm p < .05). The strongest pairs are Adventure-Animation (z 41.7), Action-Adventure, Horror-Mystery, Biography-History and Biography-Drama. The pairs that meet least often are Comedy-Thriller (z -19.4), Adventure-Drama and Action-Drama.
- **Second method.** The disparity filter keeps 24 links, 21 of them also in the null backbone. The null backbone keeps 20 links the filter drops. The filter judges a link only against the genre's own other links, so a link can beat chance and still be too weak next to the genre's strongest links to survive.
- **Communities.** Louvain on the backbone finds three genre families. The first is Drama with Biography, Comedy, History, Music, Romance, Sport and War. The second is Action with Adventure, Animation, Family, Fantasy and Sci-Fi. The third is Crime with Horror, Mystery and Thriller. Modularity is 0.39, and 20 seeds agree with an NMI of 0.77.
- **Is that structure real?** The same backbone with its links rewired (degrees kept) and its weights shuffled reaches modularity 0.40 on average. So the three families are no stronger than the genres' degrees alone produce (p 0.62). Treat them as a description, not a finding.
- **Returns.** With each film split across the families by its share of genres, no family differs from the budget-matched shuffle: -0.006, -0.019 and +0.036, all Holm p above 0.4.

The figure (`outputs/network/figures/q1_genre_network.png`) shows the result. The winners and losers of the main test sit in the same families. Animation is linked to Action and Adventure, Horror to Crime and Thriller, and Comedy to Drama through Romance. So the return effect belongs to single genres, not to the families they travel with.

## 2. Do films made by the same people share genre? (week 3)

Films are linked when they share a director, writer, producer or one of the first three billed actors. That gives 6,346 films, 18,868 people and 180,396 links, with a mean degree of 57. For each genre, the assortativity coefficient of its 0/1 indicator is compared with the genre sets shuffled between films of the same era (1,000 shuffles, each film keeps its links).

- 72 % of links join two films that share a genre, against 56 % after shuffling (p 0.001).
- All 18 genres are assortative. The strongest are Horror (0.26), Comedy (0.24), Action (0.20), Animation (0.20) and Drama (0.18). The weakest are War (0.01) and Music (0.02). Every one is far outside the shuffle, whose 95 % band ends below 0.007.

This is where Q1 and Q2 meet. The people around a film stick to genres, Horror most of all, so part of what a film-network measure picks up can be genre. Niklas's Q2 plans an NMI between his communities and the genres. These numbers are the genre side of that.

## 3. Other ways to say what genre a film is (weeks 6, 7)

IMDb gives up to three unweighted tags. Three other descriptions are already on disk:

- **Tag Genome shares** give each film a share per genre, from what MovieLens users tagged (4,231 of the films).
- **Plot shares** are the same shares predicted from the Wikipedia plot's embedding (6,303 films).
- **Plot clusters** come from k-means on the plot embeddings with k = 18, the number of kept genres. These are genres made from text alone. Ten seeds agree with an NMI of 0.70.

How well they agree:
- **Dominant genre against the IMDb tags.** A film's dominant Tag Genome genre is one of its IMDb genres for 86.9 % of films, against 24.8 % if the same genres went to random films. For the dominant plot genre it's 78.6 % against 32.5 %.
- **Plot clusters against both.** The clusters agree only loosely with either (NMI 0.26 with Tag Genome, 0.29 with plot shares). Text alone groups films by setting and story more than by genre.

Which predicts returns best? These are 5-fold cross-validated ridge regressions, repeated 5 times, on the 4,203 films that have every description. The table shows the R² each one adds on top of budget decile and era (0.094):

| Added | CV R² gained |
|---|---|
| IMDb tags (0/1) | +0.016 |
| Tag Genome shares | +0.032 |
| plot shares | +0.025 |
| plot clusters | +0.001 |
| plot embedding (384 numbers) | +0.048 |
| IMDb tags + Tag Genome shares | +0.039 |

- **Weighted genre helps.** Weighted genre doubles what the 0/1 tags explain: knowing a film is mostly Comedy helps more than knowing it has a Comedy tag.
- **Plot clusters add nothing.** Genres made from text alone, without labels, don't follow returns.
- **The plot embedding needs a caveat.** It adds the most, but a Wikipedia plot is written after release and the embedding has 384 numbers. Part of it can be fame rather than content, which is the same leak the repo's plot-tag check (`scripts/reviews/plot_tags.py`) warns about.

On the Tag Genome films (`outputs/shares/share_tests.csv`), counting each film by its shares keeps the main signs for Animation, Family, Drama and Crime. Horror weakens here and stops passing: +0.111 with shares, but also +0.134 with its plain IMDb tags (Holm p 0.34) on these same films. So the subset, which leans to much-tagged films, is what weakens it, not the shares.

## 4. Inside Horror: which plot words go with beating the budget (week 6)

Horror films with a Wikipedia plot (815) are split by their return net of budget decile and era, top quarter (204 films) against bottom quarter (204). Their plots are compared with the log-odds ratio with an informative Dirichlet prior, the same statistic Q3 uses on scripts, with the prior taken from all usable-money plots. The labels are shuffled 1,000 times within budget deciles. Words that are mostly capitalised mid-sentence are dropped as names. That drops 38,192 types, and a few real words with it, such as "new" (as in New York).

- **Horror that beats its budget** is supernatural: demon, paranormal, possessed, haunting, demonic, footage, supernatural, occult, possessing, house, night, voice, family.
- **Horror that doesn't** is physical: fight, human, run, hold, mountains, and also rape. Function words (him, upon, of, off, this) show up as well, which looks like a difference in how the plots are written rather than in the stories.
- **Survives the shuffle.** The strongest single word (|calibrated z| 4.32) beats the family-wise shuffle (p 0.004), and the top 5, 10 and 20 words on both sides beat it too (p 0.001).
- **Not plot length.** Median plot length is about the same on both sides (622 and 645 words).

So the Horror that pays off is the possession and haunting kind, which fits the $5-30M Horror cells in the tier heatmap. That's an observation about plots as Wikipedia editors describe them after release, not a cause. Supernatural horror may also be what certain producers make, which is part 2's point again.

## 5. The genre distribution through the same analyses

`q1_share_followups.py` reruns the follow-ups with the Tag Genome genre shares from `notebooks/tag_shares.ipynb` as weights, next to the 0/1 tags on the same 4,231 films (the Tag Genome stops in 2014). Outputs are in `outputs/share_followups/`, and the notebook's exploratory part shows them. It ran in 2.5 min on the HPC.

- **No audience-size leak in the shares.** `tag_profit.ipynb` found that the content tags predict IMDb votes very well, so the first check was whether the genre shares do too. They don't: they add +0.018 to the CV R² of ln(votes), against +0.022 for the tags. For the return they add +0.030 against +0.016, and with ln(votes) already in the model +0.054 against +0.033. So the shares don't carry audience size. They do carry reception, though, which part 6 found. On their own they predict IMDb rating much better than the tags do.
- **Budget tiers.** With shares, 9 of 48 tier cells pass, against 5 of 59 with tags on these films. All nine are in the $5-30M and $30-100M tiers again: Family (+0.48 and +0.32), Romance, Comedy and Animation above, Sci-Fi (-0.39), Crime and Drama below.
- **Smooth curve and hit share.** With shares the clearest winners are Family (+0.349), Romance and Comedy, and the clearest losers Sci-Fi (-0.281), War, Crime, Fantasy and Drama. Animation and Horror are weaker on these older films with either measure. A film that is mostly Family has a hit 9.4 points more often than expected.
- **Genre mix.** The two measures of a mixed film point opposite ways. Films with one IMDb genre beat their budgets (+0.238), and films with three fall short (-0.052). But the most mixed fifth by Tag Genome shares beats its budget (+0.296). The Tag Genome spread goes with votes (Spearman 0.17 net of budget), and IMDb's count of genres doesn't (-0.03). So a broad Tag Genome mix partly marks a widely seen film, as `tag_profit.ipynb` found, and IMDb's count is the cleaner measure. On it, single-genre films do best.
- **Share network.** Soft co-occurrence against a null that shuffles each film's shares across the genres gives 34 backbone links, 22 of them shared with the 41 tag links. Louvain gives five families with an NMI of 0.41 against the tag families. Horror moves to the Action and Sci-Fi family, and Animation and Family join Comedy and Romance. Modularity is low (0.14), so the families are loose here too.
- **Assortativity.** All 18 genres are assortative with shares too, and more strongly for most genres. Horror is 0.27 against 0.17 with tags on the same films, Action 0.25 against 0.19, and Family 0.15 against 0.10. The people around a film stick to how much of a genre it is, not just to the tag.
- **Horror words, three ways.** On every IMDb Horror film (815) the supernatural words lead, as in part 4. On IMDb Horror films with Tag Genome scores (441), and on films whose largest share is Horror (351), the two lists agree with each other: receives, walls, murders and footage above, human, fight and upon below. Neither has demon or possessed near the top. Since (b) and (c) agree, the definition of Horror isn't what changes the list. The smaller and older set of films is. On those sets the strongest single word no longer beats the family-wise shuffle (p 0.04 and 0.08), though most of the list-level tests still pass. The below-budget words (human, fight, upon) are stable in all three.
- **Is supernatural Horror only a recent thing?** No. Counting a film as supernatural when its plot has any of demon, demonic, possessed, possessing, paranormal, haunting, supernatural or occult, about as many Horror films are supernatural before 2015 (26 %) as from 2015 on (28 %). They beat other Horror films on return net of budget and era in both periods: by +0.44 before 2015 (p 0.008) and by +0.81 from 2015 on (p 0.001). The gap is larger recently, but it isn't new. This check is post hoc, since the words were picked from the full list, which includes these films (`outputs/words/horror_supernatural_by_period.csv`).

## 6. Fixing the genre distribution's weaknesses

A critical look at the shares found six problems. `q1_share_fixes.py` fixes each one, choosing every setting from agreement with IMDb's labels, never from returns. Outputs are in `outputs/share_fixes/` and the notebook's exploratory part. It runs in 9 s.

- **One tag per genre.** For each genre a fixed set of candidate tags, and their mean, is scored by AUC against IMDb's label. Six genres get a better scorer, all as means of their candidates. Family goes from 0.879 (kids and family) to 0.928 (mean of kids and family, family, kids, children). Biography, Comedy, History, Romance and Thriller gain less.
- **The arbitrary power.** Choosing p by how well each share reproduces IMDb's labels (cross-validated log loss) gives p = 1.5 instead of 2. The return gain still depends on p, falling from +0.064 at p = 0.5 to +0.022 at p = 4. That's why p is chosen from the labels, and the full table is in `cv_r2.csv`.
- **Total tagging is fame.** The log of a film's total genre relevance predicts ln(votes) far better (+0.106) than the shares do (+0.027), and it adds +0.055 to the return on its own. Shares plus log total add +0.129, so most of the raw scores' big number (+0.128) is how much the film is tagged, not its genre. The log total is now its own variable and the shares are pure composition.
- **Reception.** The shares predict IMDb rating better than the tags (+0.264 against +0.191). Holding rating and ln(votes) fixed in the model, the chosen shares still add +0.065 to the return against +0.030 for the tags. Adding Rotten Tomatoes on the films that have it gives +0.069 against +0.024. So part of the shares' edge was reception, and an edge remains without it. Residualising the shares on rating and votes and then leaving those out of the model looks like a fix but isn't: a linear model can rebuild rating and votes from the 18 residual columns, which made it score +0.065 with no reception in the base.
- **Genre effects.** The order of genre effects with shares agrees with the tags' order at Spearman 0.60 to 0.76, depending on what is held fixed. Family and Romance lead and Sci-Fi, War and Crime trail. Family and Animation shares overlap so much that the Animation share turns negative next to Family. Read share effects as an order, not as sizes.
- **Coverage.** On the Tag Genome subset every genre looks better than on the full set (mean gap 0.139), but the order of genres holds (correlation 0.97). Reweighting the subset by decade × vote tier × budget decile shrinks the gap to 0.093. 9 % of the full set sits in cells with no scored film at all (mostly after 2014), so the gap can't fully close.
- **The genre mix was fame.** Alone, a more mixed film beats its budget (+0.129 per SD). With the log total held fixed the sign flips (-0.157), and with rating and votes also held fixed it stays negative (-0.117). That matches IMDb's count of genres (-0.122 per SD). So more mixed films earn less, and part 5's "most mixed earns most" came from fame.
- **Before release: genre shares from the screenplay.** A ridge regression from each script's embedding to its Tag Genome shares, out-of-fold, makes a genre distribution that exists before release and can't contain ratings. It's a valid measure of genre: mean AUC 0.90 against IMDb labels, a median correlation of 0.65 with the Tag Genome shares, and the dominant genre is one of the IMDb genres for 81 % of films. MiniLM and Word2Vec vectors give almost the same numbers. It carries less reception than the tags: +0.117 to IMDb rating against +0.179. But it predicts returns no better than the tags on the common set: +0.023 against +0.025 with rating and votes fixed, and +0.030 together. In the budget-matched shuffle no genre passes with script shares, because predicted shares are smoothed.
- **Hand check, for the group.** 562 films have a dominant share that is none of their IMDb genres. A random 40 are in `outputs/share_fixes/hand_check_sheet.csv` with titles, years, IMDb genres and the top three shares. Each needs someone to read the synopsis and write in `which_is_right` whether IMDb, the Tag Genome, both or neither names the genre better. The notebook counts the answers as they come in. I haven't labelled any: it's meant as a human check.

The fixed shares hold up better than the originals, and the claim survives in a smaller form. With reception held fixed, weighted genre adds about twice what IMDb's tags do (+0.065 against +0.030). The before-release version from scripts matches the tags but doesn't beat them.

## 7. Horror words in the screenplays

`q1_horror_scripts.py` runs the word test of part 4 on the screenplays. On the 203 Horror films with a script, no single word survives the family-wise shuffle in either layer (p 0.40 description, 0.21 dialogue). The below-budget side of the description layer does pass as a list (k = 10, p 0.009). Two thirds of Horror scripts use at least one of the supernatural words somewhere, so counting them doesn't separate the films (+0.17, p 0.35). So the Wikipedia-plot finding may say more about how plots are summarised after release than about the scripts. With 203 films this test is also much weaker than the plot test on 815.
