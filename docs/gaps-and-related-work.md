# Gaps and related work (15 November 2026)

This file lists what's missing from the group's Google Doc ("Social Graphs Final Project") and from the project as a whole, checked against the course brief in [project-context.md](project-context.md). It then lists published papers that did something close to what we do. The paper is due Friday 27 November and the presentation is Wednesday 2 December.

I read the doc's text through the Docs API. That API doesn't return comment threads, so any comments in the doc aren't covered here.

## 1. Against the course rubric

These matter most, because they're what gets graded. They're ordered by rubric category.

**Analysis and checks.** The rubric asks for "a null, a second method, or a hand-labeled sample, with the result reported". We have nulls and second methods everywhere, but no hand-labelled sample anywhere yet.

- Review sentiment: [main.tex:107](../paper/main.tex) still says "TODO: hand-labelled sample of 50 reviews". The doc says the same ("hand-labelling 50 reviews ... I'd like us to split that"), with no owner.
- Screenplay arcs: dropped on 17 November (kept on the `archive/scripts-arcs` branch), so their hand-label sheets no longer need filling.

If we only finish one thing this week, I think it should be the review label set, because it takes a few hours and closes a rubric line.

**Paper front matter.** Still TODO in [main.tex](../paper/main.tex):

- line 24: author contributions (the course wants specific roles) and who did the hand labelling
- line 26: corresponding e-mail
- line 92: data collection dates (they're in [imdb-data-structure.md](imdb-data-structure.md): IMDb snapshot 4 Nov, Wikipedia pull 4 Nov, Wikidata at least 31 Oct)
- lines 110 and 115: public repo link
- line 117: acknowledgements

**Numbers that went stale.** The 15 November infobox fixes moved the clean sample from 6,851 to 6,843 ([TESTING.md](../TESTING.md), line 82), but [main.tex:57](../paper/main.tex) (Fig. 1 caption) and [main.tex:92](../paper/main.tex) (Methods) still say 6,851. The README's similarity section already uses 6,843. The reception numbers also need a look: Results and the Fig. 3 caption (lines 76 and 81) say 2,095 films, while Methods (line 92) says 2,070. These may be two different subsets, but the paper should say which. AGENTS.md asks for every number to come from a script's output, so these should be re-read from the current outputs, not edited by hand.

**The 10k+ tier isn't hand-checked.** The README says that tier's parsing "isn't hand-checked yet", and TESTING.md records parser fixes but no read-through of sample rows. The similarity result in Fig. 1B and the prediction model both run on it. AGENTS.md says every parser bug so far was found by reading rows, so reading 30 to 50 rows from the 10k-100k tier before hand-in would protect the headline result.

**Showcasing the course: the network half is thin on network science.** The pre-release signal comes mostly from track records, and the paper reports that PageRank adds nothing beyond experience (ρ = +0.002, line 50). That's an honest result, but it leaves little that looks like network science from the course. Methods line 95 describes Louvain communities on the company co-credit projection, yet no community result appears in Results. Also, the contributions line credits N.J. with "the actor co-appearance network and its interactive explorer", and that network isn't in Results either. The group should decide whether it goes in (for example as the figure that introduces the graph), or whether the contributions line changes.

**References.** 21 of the 28 entries in [references.bib](../paper/references.bib) are `checked = {abstract}`, only 7 `read`. The literature overview says to check those against the papers before citing any number from them. Items 2 and 3 of its "Worth adding before the paper" list are still open (Berger's volatility as an exploratory arc descriptor, and IMDb vote count as review volume). Item 4, the Holbrook sign flip, is already in the Discussion (line 86).

**Length.** Six pages including about 36 references in PNAS two-column. The class files live on Overleaf, so this can only be checked there.

**Slides and reflection.** I found no slide or reflection files in the repo. The three-page slide PDF and the half-page reflection are both due with the paper on 27 November. They may live on Overleaf or elsewhere. If not, they're still to do.

## 2. The Google Doc against where the project is

The rubric says everyone must be able to explain everything in the paper, so the shared doc going stale is a real risk on presentation day. The paper draft (14 November) has moved well past it.

- **The proposals are still open in the doc.** The doc asks the group to pick A (more data, more flops), B (focus on scripts) or C (keep reviews as an outcome). The paper has in effect done all three: it uses the 10k+ tier (A), has a full screenplay section (B) and treats reception as an outcome (C). The doc should record that decision and why.
- **"No network link yet"** is out of date. The network results (track records, return similarity across shared people, movers, team history) are the paper's main finding.
- **"What is success for us?"** is asked and never answered in the doc. The paper's answer is ln(gross / budget), with gross minus budget rejected because it rewards blockbusters (README, "Decisions still open"). The doc should say so, with the caveat that neither is real profit.
- **The methods table has one row** (sentiment analysis of reviews). The paper uses about five methods that each need a "we chose X over Y because" line: ridge with an out-of-time split, the shuffle-null similarity test, Fightin' Words log-odds, VADER on reviews, and Louvain. Filling this table is also good preparation for the questions on 2 December.
- **"Reviews: Kaggle"** no longer matches. The audience reviews are the MovieLens Tag Genome and critic text comes from Wikipedia.
- **Network design.** The doc says nodes are films, linked if they share actors, directors or studios. The built graph is bipartite (films, people, companies) with projections, which the doc should describe.
- **The 11 November TODOs** (Angi on reviews, Gyula on screenplays, Niklas on the network) have no status. Each has a section in the paper now, except Niklas's actor network (see above).
- **Ideas left unused.** Weekly reviews against weekly revenue can't be done, because The Numbers and Box Office Mojo forbid scraping (AGENTS.md). Review embeddings against profit are in [scripts_and_reviews.ipynb](../notebooks/scripts_and_reviews.ipynb) but not in the paper. The doc should mark both as dropped or parked, so nobody raises them as open work.
- **The flop count** in the doc (130 of 2,095 lose money) is from the 100k+ reception sample. The README's corresponding figure is 169 of 2,446 for the 100k+ clean sample, and 38.1% of the films in the 10k to 100k tier gross under budget. Worth one line so the two counts aren't confused.

## 3. Papers that did something close to ours

The six recommended papers below are now in `references.bib` with `checked = {metadata}`, so read at least the abstract before citing one. `python3 scripts/get_papers.py` downloads whatever is open access to `data/raw/papers/` and prints DTU Findit links for the rest. Status follows the bib's convention, with two extra levels below `abstract`:

- `doi`: I checked title, authors, venue, issue and pages on Crossref (15 Nov).
- `found`: located by search (DOI, arXiv, ACL Anthology or publisher page), with the summary taken from abstracts and secondary sources, not the full text.

### What this means for the Introduction

The search didn't turn up any paper that builds network or track-record features only from earlier films and tests them against return on budget. It also found nothing on return correlation between films that share people, or on whether a person's record carries over when they move studio. That supports the framing in [main.tex:43](../paper/main.tex). It's a gap in a limited search, not proof nothing exists, so the paper should say "we know of no study that..." at most, and not "first".

Two papers could push back on our claims, so read them before hand-in:

- **Paulich & Kumar 2021** relate screenplay features to box office, and a search snippet says they also look at ROI. If they find a script effect on ROI, the title "what its script says does not" needs a sentence on why we differ (franchise handling, genre × decade controls, sample).
- **Hofmann et al. 2017** is a meta-analysis showing star effects are inflated when selection is ignored. It backs our leakage-safe design and qualifies Elberse 2007, which we cite.

### Recommended for the paper (pick 3 to 5)

Six pages leave room for only a few more references. These are the ones that change what the paper can say:

| Paper | Why it earns a citation | Status |
|---|---|---|
| Viana et al. 2014 | Closest network paper: IMDb director, producer and writer network over time, metrics against gross and rating. Lets us say what we add (return, studios, cast, earlier-films-only) | `doi` |
| Hofmann et al. 2017 | Star effects inflated by selection; supports our design | `found` |
| Paulich & Kumar 2021 | The post-2016 script-to-money paper our null result has to answer | `doi` |
| Pokorny & Sedgwick 2010 | One of the few papers with profitability, not revenue, as the outcome | `found` |
| Delmestri et al. 2005 | Separates commercial success from artistic merit using network ties, like our Results section 3 | `found` |
| Wasserman et al. 2015 | PNAS, compares critics, IMDb ratings and a film citation network as quality measures; fits the reception section and the journal | `doi` |

### Reading list by piece

**Networks and track records**

| Paper | What they did | How ours differs | Status |
|---|---|---|---|
| Viana, Santos, Couto da Silva, Moro 2014, LA-WEB, [10.1109/LAWeb.2014.10](https://doi.org/10.1109/LAWeb.2014.10) | IMDb network of directors, producers, writers over time; metrics against gross and rating | Gross not return; no studios or actors | `doi` |
| Zhang, Tang, Xiong, Wang, Zhang 2019, *Entropy* 21(9):859, [10.3390/e21090859](https://doi.org/10.3390/e21090859) | Alliance network of production and distribution companies, 903 films; degree and structural holes against box office | Companies only; revenue | `doi` |
| Meiseberg & Ehrmann 2013, *J. Cultural Economics* 37(1):61-86, [10.1007/s10824-012-9173-7](https://doi.org/10.1007/s10824-012-9173-7) | Team diversity and domestic vs foreign success; degree in the people-film network | Diversity and degree, not track record | `found` |
| Rossman, Esparza, Bonacich 2010, *ASR* 75(1):31-51, [10.1177/0003122409359164](https://doi.org/10.1177/0003122409359164) | Centrality in the credit network, experience and team spillovers on Oscar nominations | Awards, not money | `found` |
| Juhász, Tóth, Lengyel 2020, *PLoS ONE* 15(2):e0229436, [10.1371/journal.pone.0229436](https://doi.org/10.1371/journal.pone.0229436) | Hungarian films 1990-2009; core creators who broker the periphery win more awards | Awards; small national industry | `doi` |
| Delmestri, Montanari, Usai 2005, *J. Management Studies* 42(5):975-1002, [10.1111/j.1467-6486.2005.00529.x](https://doi.org/10.1111/j.1467-6486.2005.00529.x) | Director ties and reputation predict commercial success vs artistic merit | Italian independents; director ego network only | `found` |
| Perretti & Negro 2007, *J. Organizational Behavior* 28:563-586, [10.1002/job.464](https://doi.org/10.1002/job.464) | Newcomer/old-timer mix and repeat teams drive genre innovation, 6,446 studio-era films | Innovation, not return | `found` |
| Williams, Lacasa, Latora 2019, *Nature Communications* 10:2256, [10.1038/s41467-019-10213-0](https://doi.org/10.1038/s41467-019-10213-0) | IMDb careers of about 2.4M actors; rich-get-richer and hot streaks | Success is steady work, not film return | `found` |
| Hofmann, Clement, Völckner, Hennig-Thurau 2017, *IJRM* 34(2):442-461, [10.1016/j.ijresmar.2016.08.006](https://doi.org/10.1016/j.ijresmar.2016.08.006) | Meta-analysis of 61 studies plus 1,545 films; star effects biased upward without selection control | Stars only | `found` |
| Wan, Ouyang, Kaplan, Han 2015, SIAM SDM, pp. 918-926 | Meta-path regression on a heterogeneous IMDb graph, box office as one target | Methods paper; revenue; no time ordering mentioned | `found` |
| Krauss et al. 2008, ECIS, [aisel.aisnet.org/ecis2008/116](https://aisel.aisnet.org/ecis2008/116) | Network and sentiment of IMDb forum posts to predict Oscars and box office | Network of forum users, not crew; post-release buzz | `found` |
| Miller 2011, *Int. J. Communication* 5:1014-1033 | Co-production company networks of top-grossing vs critically lauded films | Descriptive, about five years | `found` |
| Lutter 2015, *ASR* 80(2):329-358, [10.1177/0003122414568788](https://doi.org/10.1177/0003122414568788) | Network closure and women's careers, about 100k US film workers | Careers, not films | `found` |

**Screenplays, arcs and character networks**

| Paper | What they did | How ours differs | Status |
|---|---|---|---|
| Paulich & Kumar 2021, *J. Academy of Marketing Science* 49(6):1222-1242, [10.1007/s11747-021-00794-y](https://doi.org/10.1007/s11747-021-00794-y) | Imagery and emotion features in screenplays against box office, possibly ROI | Abstract not retrieved; sample and method unverified | `doi` |
| Lindner, Lindquist, Arnold 2015, *Sociological Inquiry* 85(3):407-428, [10.1111/soin.12081](https://doi.org/10.1111/soin.12081) | Bechdel-passing films earn less domestically, net of genre, critics, sequel and stars | Gender, not script text in general | `doi` |
| Chu & Roy 2017, IEEE ICDM, [10.1109/ICDM.2017.100](https://doi.org/10.1109/ICDM.2017.100) | Audio and visual emotional arcs clustered into shapes; predict engagement on web shorts | Arcs from audio/video; engagement, not return | `found` |
| Gorinski & Lapata 2015, NAACL, [10.3115/v1/N15-1113](https://doi.org/10.3115/v1/N15-1113) | Scene graphs from scripts (character interaction plus sentiment) for summarisation | No commercial outcome | `found` |
| Agarwal et al. 2015, NAACL, [10.3115/v1/N15-1084](https://doi.org/10.3115/v1/N15-1084) | Automated Bechdel test from screenplays | No money outcome | `found` |
| Agarwal et al. 2014, CLFL workshop, ACL Anthology W14-0907 | Character networks parsed from screenplays | No money outcome | `found` |
| Kagan, Chesney, Fire 2020, *Palgrave Communications* 6:92, [10.1057/s41599-020-0436-1](https://doi.org/10.1057/s41599-020-0436-1) | 15,540 character networks from subtitles plus IMDb; gender and centrality over time | No financial outcome | `found` |
| Labatut & Bost 2019, *ACM Computing Surveys* 52(5):89, [10.1145/3344548](https://doi.org/10.1145/3344548) | Survey of character-network extraction | Background | `found` |
| Kaminski et al. 2018, Moviegalaxies dataset, [10.7910/DVN/T4HBA3](https://doi.org/10.7910/DVN/T4HBA3) | Character networks for films (dataset only; no peer-reviewed paper found) | Cite as data if used | `found` |

**Reception, quality and profit**

| Paper | What they did | How ours differs | Status |
|---|---|---|---|
| Wasserman, Zeng, Amaral 2015, *PNAS* 112(5):1281-1286, [10.1073/pnas.1412198112](https://doi.org/10.1073/pnas.1412198112) | Film citation network of 15,425 US films; long-gap citations predict National Film Registry entry as well as critics or IMDb ratings | Cultural significance, not profit | `doi` |
| Spitz & Horvát 2014, *PLoS ONE* 9(10):e108857, [10.1371/journal.pone.0108857](https://doi.org/10.1371/journal.pone.0108857) | Centrality on a reference network of 40k+ films as long-term impact | No money outcome | `found` |
| Wallisch & Whritner 2017, *Projections* 11(1), [10.3167/proj.2017.110107](https://doi.org/10.3167/proj.2017.110107) | About 3,000 people rating about 200 films agree little with each other and weakly with critics | No profit outcome | `found` |
| Gemser, Van Oostrum, Leenders 2007, *J. Cultural Economics* 31(1):43-63, [10.1007/s10824-006-9025-4](https://doi.org/10.1007/s10824-006-9025-4) | Review effects on art-house vs mainstream box office (direction unconfirmed) | Dutch data; revenue | `doi` |
| Pokorny & Sedgwick 2010, *Economic History Review* 63(1):56-84, [10.1111/j.1468-0289.2009.00488.x](https://doi.org/10.1111/j.1468-0289.2009.00488.x) | Hollywood profitability 1929-1999, studio output as a portfolio | Studio ledgers; no networks | `found` |

**Box-office forecasting from attention and pre-release data**

| Paper | What they did | Status |
|---|---|---|
| Mestyán, Yasseri, Kertész 2013, *PLoS ONE* 8(8):e71226, [10.1371/journal.pone.0071226](https://doi.org/10.1371/journal.pone.0071226) | Wikipedia views and edits for 312 films predict revenue a month before release. Same data source as ours, different signal | `found` |
| Asur & Huberman 2010, WI-IAT, [10.1109/WI-IAT.2010.63](https://doi.org/10.1109/WI-IAT.2010.63) | Tweet rate and sentiment forecast box office | `found` |
| Kim, Hong, Kang 2015, *Int. J. Forecasting* 31(2):364-390, [10.1016/j.ijforecast.2014.05.006](https://doi.org/10.1016/j.ijforecast.2014.05.006) | Korean box office from social-media mentions with ML | `found` |
| Ghiassi, Lio, Moon 2015, *Expert Systems with Applications* 42(6):3176-3193, [10.1016/j.eswa.2014.11.022](https://doi.org/10.1016/j.eswa.2014.11.022) | Pre-production features in a neural network; revenue | `found` |
| Simonoff & Sparrow 2000, *Chance* 13(3):15-24 | Predicting gross from pre-release information | `found` |

**Background and further reading**

- Watts & Strogatz 1998, *Nature* 393:440-442, [10.1038/30918](https://doi.org/10.1038/30918): the film-actor graph is one of its three networks. A natural citation if the paper shows the actor network. `found`
- Barabási & Albert 1999, *Science* 286:509-512, [10.1126/science.286.5439.509](https://doi.org/10.1126/science.286.5439.509): the actor graph as a scale-free example is from memory, not confirmed. `found`
- Dadlani et al. 2024, "Leading by the nodes", *Applied Network Science* 9:76, [10.1007/s41109-024-00673-9](https://doi.org/10.1007/s41109-024-00673-9): survey of film-industry network studies. Its bibliography is the best place to look for anything this search missed. `found`

Don't cite these, because they couldn't be confirmed: an Ebbers & Wijnberg 2012 "nexus of networks" paper (only their 2009 *Human Relations* and 2010 *Strategic Organization* papers turned up), Meiseberg, Ehrmann & Dormann 2008, and any peer-reviewed GNN-on-the-IMDb-cast-graph box-office paper. The only one found is an unreviewed preprint (Shen 2020, arXiv:2006.13417). The one Reagan et al. replication found is a Leiden master's thesis (Pouli 2023), which recovers most of the six shapes with VADER and isn't citable as peer-reviewed.
