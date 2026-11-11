# Data bundles

Compressed data so the group doesn't have to re-pull or re-download it. Extract into `data/raw/` from the repo
root:

```bash
mkdir -p data/raw && for f in data/bundles/*.tar.zst; do tar --zstd -xf "$f" -C data/raw; done
```

| File | Size | What's inside | Licence |
|---|---:|---|---|
| `wikipedia-wikidata-2026-11-11.tar.zst` | 97 MB | English Wikipedia article wikitext for the 10,000+ vote movies (pulled 29 October to 4 November 2026), IMDb-to-Wikipedia links, Wikidata tables, parsed infobox tables | Wikipedia: CC BY-SA 4.0 (English Wikipedia contributors; each row keeps its page and revision ID). Wikidata: CC0 |
| `moviesum-screenplays-2026-11-11.tar.zst` | 90 MB | MovieSum screenplays (`scripts/moviesum/*.jsonl`) | CC BY-NC 4.0 (Saxena and Keller, Findings of ACL 2024); the screenplays themselves remain under the rights holders' copyright |
| `screenplay-words-2026-11-11.tar.zst` | 10 MB | Dialogue word counts per screenplay | Derived from MovieSum: non-commercial, with the credit above |
| `text-features-2026-11-11.tar.zst` | 15 MB | Screenplay and review features and embeddings (numbers only) | Derived from MovieSum and MovieLens Tag Genome 2021 (CC BY-NC 3.0): non-commercial, cite both |
| `movie-graph-2026-11-11.tar.zst` | 9 MB | The movie graph (`features/graph/`): nodes, edges, GraphML, credits, people, company IDs, Wikipedia producer matches, as built by PR #9 | Built from IMDb's non-commercial datasets (personal, non-commercial use under IMDb's terms), Wikipedia (CC BY-SA 4.0) and Wikidata (CC0) |

Not here: IMDb's raw files (1.9 GB; `python3 scripts/get_data.py imdb`) and the Tag Genome reviews (1.9 GB;
`python3 scripts/get_data.py genome`), both over GitHub's file limit.

These files are part of the git history: every clone downloads them, and removing them later would mean
rewriting history. Add a new dated bundle rather than replacing a file.
