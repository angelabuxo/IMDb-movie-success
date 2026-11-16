# Data bundles

Compressed data so the group doesn't have to re-pull or re-download it. `./setup.sh --data` from the repo root
extracts them into `data/raw/` (`./setup.sh` also sets up Python). By hand, extract oldest first, so a newer
bundle replaces the files it updates:

```bash
mkdir -p data/raw && for f in $(ls data/bundles/*.tar.zst | sed -E 's/.*-([0-9]{4}-[0-9]{2}-[0-9]{2})\.tar\.zst$/\1 &/' | sort | cut -d" " -f2); do tar --zstd -xf "$f" -C data/raw; done
```

| File | Size | What's inside | Licence |
|---|---:|---|---|
| `wikipedia-wikidata-2026-11-11.tar.zst` | 97 MB | English Wikipedia article wikitext for the 10,000+ vote movies (pulled 29 October to 4 November 2026), IMDb-to-Wikipedia links, Wikidata tables, parsed infobox tables | Wikipedia: CC BY-SA 4.0 (English Wikipedia contributors; each row keeps its page and revision ID). Wikidata: CC0 |
| `moviesum-screenplays-2026-11-11.tar.zst` | 90 MB | MovieSum screenplays (`scripts/moviesum/*.jsonl`) | CC BY-NC 4.0 (Saxena and Keller, Findings of ACL 2024); the screenplays themselves remain under the rights holders' copyright |
| `screenplay-words-2026-11-11.tar.zst` | 10 MB | Dialogue word counts per screenplay | Derived from MovieSum: non-commercial, with the credit above |
| `text-features-2026-11-11.tar.zst` | 15 MB | Screenplay and review features and embeddings (numbers only) | Derived from MovieSum and MovieLens Tag Genome 2021 (CC BY-NC 3.0): non-commercial, cite both |
| `movie-graph-2026-11-11.tar.zst` | 9 MB | The movie graph (`features/graph/`): nodes, edges, GraphML, credits, people, company IDs, Wikipedia producer matches, as built by PR #9 | Built from IMDb's non-commercial datasets (personal, non-commercial use under IMDb's terms), Wikipedia (CC BY-SA 4.0) and Wikidata (CC0) |
| `movie-graph-2026-11-15.tar.zst` | 12 MB | The movie graph, the parsed infobox tables (`features/wiki_infobox*.csv`), the movie tables (`features/movies_features*.csv`) and the network features (`film_network_features.csv`, `roi_by_entity.csv`) after the 15 November parser fixes: names split or recovered from lists, and currencies before `$` read correctly. It replaces the same files from `wikipedia-wikidata-2026-11-11` and `movie-graph-2026-11-11` | Same as `movie-graph-2026-11-11` |
| `movie-graph-2026-11-16.tar.zst` | 7 MB | `features/graph/nodes.csv`, `features/graph/movie_graph.graphml`, the parsed infobox tables (`features/wiki_infobox*.csv`) and the movie tables (`features/movies_features*.csv`) after the tightened clean rule of 15 November: films before 1975, direct-to-video films and grosses whose citation says rentals are dropped, leaving 6,413 clean movies. Only the `clean` and `gross_flag` columns differ from `movie-graph-2026-11-15`, whose six files of the same name it replaces. `film_network_features.csv` and `roi_by_entity.csv` aren't rebuilt and still come from the 6,843-movie sample | Same as `movie-graph-2026-11-11` |
| `extra-tables-2026-11-15.tar.zst` | 0.5 MB | Small tables no other bundle holds: franchise and sequel flags (`features/franchise.csv`), MovieSum IDs and screenplay statistics (`moviesum_ids.csv`, `moviesum_stats.csv`) and Tag Genome review counts (`genome_review_counts.csv`). Numbers and IDs only | Wikidata: CC0. MovieSum and Tag Genome counts: non-commercial, with the credits above |
| `tag-features-2026-11-16.tar.zst` | 46 MB | Per-film tag scores and plot embeddings (numbers only): Tag Genome relevance for 1,084 tags (`features/tag_relevance*`), genre shares (`genre_shares.csv`), each film's most distinctive tags (`tag_top.csv`), per-tag statistics (`tag_stats.csv`), the Wikipedia plot embeddings (`plot_vectors.npy`, `plot_vector_ids.csv`) and the genre shares predicted from plots (`plot_genre_shares.csv`). Built by `scripts/reviews/tag_shares.py`, `plot_embeddings.py` and `plot_tags.py` on the 2026-11-15 movie graph. Leaves out `plot_tag_relevance.npy` (53 MB), which `plot_tags.py` rebuilds from these files | Tag Genome 2021 CC BY-NC 3.0: non-commercial, cite both papers in its readme. Plot embeddings derived from English Wikipedia (CC BY-SA 4.0) |

Not here: IMDb's raw files (1.9 GB; `python3 scripts/get_data.py imdb`) and the Tag Genome reviews and scores (1.9 GB;
`python3 scripts/get_data.py genome`), both over GitHub's file limit. The tag scores the project uses are in
`tag-features-2026-11-16`, so the Tag Genome download is only needed to rebuild them.

These files are part of the git history: every clone downloads them, and removing them later would mean
rewriting history. Add a new dated bundle rather than replacing a file.
