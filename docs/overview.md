# Project overview

Log-Log Legends, DTU 02805, autumn 2026. State on 14 November. Paper due 27 November, presentation
2 December. More detail: [project-details.md](project-details.md).

**Question:** what makes a movie highly profitable, and does making money agree with being liked?

**Profit** is ln(worldwide gross / budget), "the multiple". It leaves out marketing and the cinemas' share, so
it's a return on the reported budget, not real profit. **Sample:** 2,446 clean films with 100,000+ IMDb
votes, or 6,851 with 10,000+.

## Three strands, one model

```
   DATA: IMDb · Wikipedia · Wikidata · MovieSum screenplays · Tag Genome reviews
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
  WHO MADE IT                  WHAT IT SAYS                  HOW IT WAS RECEIVED
  movie graph                  Scattertext words             critics vs audience
  actor network                emotional arcs                vs money
  shared-people returns        script features
       └────────────────────────────┼────────────────────────────┘
                                    ▼
          WHAT PREDICTS PROFIT: train on films to 2010, test on 2011-2017
```

## What we've found

| Strand | Result |
|---|---|
| Who made it | The best predictor known before release. Studio and people track records lift out-of-sample R² from −0.01 to 0.20. Films that share people have similar returns. |
| What it says | Almost nothing once budget, genre and era are fixed. The word difference vanishes without franchise films, and the arcs show no clear link (one near miss). |
| How it was received | Money and liking agree only moderately (ρ 0.27 to 0.33). Reviews add to the model (R² 0.24 to 0.31), but they come after release. |

**Possible one story:** who makes a film predicts its return, what's in the script doesn't, and audiences
only partly reward what earns.

## Rules every strand follows

- Network measures use only earlier films.
- Every result gets a check: a shuffle null, a second method, or hand labels.
- Budget, era and genre are controlled, and franchise films get a separate check.
- No script text, review text or IMDb fields go into git.

## Open

- Agree on the one story for the paper.
- Fill the human validation sheets (Scattertext, arcs, reviews).
- Merge Àngela's `reviews-exploration` and Niklas's `initial_actor_network`.
- Arcs Step 11 (script tone vs review tone) needs the group to decide whether whole-review VADER counts as the
  same instrument.
