# What goes with a film earning back its budget?

Final project for DTU 02805 Social Graphs and Interactions, autumn 2026, by Log-Log Legends: Gyula Kürthy, Àngela Buxó and Niklas Johansen.

The paper is [`paper/main.tex`](paper/main.tex). It asks which of three things known before release goes with a film's return on budget, ln(gross / budget): its genre, the people who make it, or its screenplay.

- **Genre.** Against films of the same budget, Animation, Family and Horror return more, and Drama, Action, Crime, History and War return less. Genre adds only 0.03 to the explained variance.
- **People.** Films that share a director or producer have more similar returns than shuffles allow (0.09 against at most 0.04). The network's communities are production houses.
- **Screenplay.** Among films of the same genre, era and budget, hit and flop scripts differ more than shuffles allow only in Horror's scene descriptions. That test is exploratory.

None of this is causal. Gross over budget isn't profit, and every film here has at least 10,000 IMDb votes, so films nobody noticed are missing.

## Where things are

| Folder | What's in it |
|---|---|
| [`notebooks/`](notebooks/) | What we did, one notebook per part of the paper. They read committed results, so they run without the raw data |
| [`src/`](src/) | The code: [`analysis/`](src/analysis/) (the tests and their outputs), [`scripts/`](src/scripts/) (getting and parsing the data), [`explorer/`](src/explorer/) (a local app) and [`tests/`](src/tests/) |
| [`data/`](data/) | Compressed data in `bundles/`, small tables, [where everything comes from](data/SOURCES.md) and [which films count](data/ANALYSIS_SET.md) |
| [`paper/`](paper/) | The paper, its references and figures |
| [`archive/`](archive/) | Analyses and notes the paper doesn't use. Some of it is still read by the explorer |

Which films count is set in one place, [`src/analysis/clean_rule.py`](src/analysis/clean_rule.py). [`src/tests/test_paper_numbers.py`](src/tests/test_paper_numbers.py) compares the numbers in the paper with the committed results and fails when one no longer matches.

Choices made before looking at results are written down in [`DECISIONS.md`](src/analysis/screenplays/DECISIONS.md) (screenplays) and [`Q2_NETWORK.md`](src/analysis/network/Q2_NETWORK.md) (people).

## Run it

```bash
./setup.sh
```

This makes `.venv`, installs the packages and unpacks the bundles into `data/raw/`. It needs `zstd`.

```bash
.venv/bin/python -m pytest
```

```bash
./src/explorer/run.sh
```

The first runs the tests, the second starts the explorer on <http://localhost:8501>. It only listens on your machine, because `data/raw/` holds licensed data and copyrighted script text, so don't deploy it. `./run_all.sh` does the setup, checks every explorer page, runs the tests and starts it.

Commands in the code's own notes, such as `python -m analysis.genre.q1_genre`, are meant to be run from `src/`.

The larger data isn't in git. [`data/SOURCES.md`](data/SOURCES.md) and [`AGENTS.md`](AGENTS.md) say how to get it and under which terms.

The README from before the November 2026 cleanup, with the full notes on every bundle and branch, is [`archive/docs/README-before-cleanup.md`](archive/docs/README-before-cleanup.md).
