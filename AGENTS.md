# Instructions for coding agents

This repo is a DTU 02805 final project: what predicts a movie's profit, and does profit agree with whether people liked the film? Start with [README.md](README.md). The course brief is summarised in [docs/project-context.md](docs/project-context.md).

## Get the data

The data isn't in git. If someone asks you to download or set up the data, run these from the repo root:

```bash
python3 scripts/get_data.py --list
python3 scripts/get_data.py --status
```

The first shows each dataset with its size and terms. The second shows what's already here. Tell the person the size before a large download, then fetch what they asked for:

```bash
python3 scripts/get_data.py imdb moviesum
```

`--all` fetches every file and Wikidata table and builds the movie lists, which comes to about 4.5 GB. Wikipedia articles are separate because the pull takes about an hour. It resumes if it stops, so run it in the background and run it again after a crash:

```bash
python3 scripts/get_data.py wikipedia
```

Everything lands in `data/raw/`, which git ignores. Use `--data-dir` to put it somewhere else. The script needs only Python 3, and it uses `aria2c` if that's installed. `aria2c` is much faster for the large files.

To parse the infoboxes, install `requirements.txt` in a virtual environment and run the script from the data folder:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cd data/raw && mkdir -p features && ../../.venv/bin/python ../../scripts/imdb/parse_infobox.py
```

## What the script can't fetch

- OpusData is a paid extract, so a person has to arrange it with sales@opusdata.com.
- Don't download the TMDB Kaggle file unless the group has settled TMDB's terms, which bar use in connection with machine learning applications.
- Don't scrape IMDb pages, Rotten Tomatoes, Letterboxd, The Numbers or Box Office Mojo. Their terms or robots rules forbid it.

## Rules

- Never commit anything from `data/raw/`, or any article text, review text or screenplay. Commit only code, docs and derived numbers.
- Don't quote long passages from screenplays or reviews, in the paper or in chat.
- Never put an email address, password or token in code.
- Don't invent a result, source or number. Every number in a doc should come from a script's output.
- Say what each measurement captures and what it misses, and check it with a null, a second method or a hand-labelled sample. The course grades this.
- Work on a branch and open a pull request. Don't push straight to `main`.

## Known data traps

- Budget and gross are parsed from Wikipedia infoboxes. Read a sample of raw rows after changing the parser, because every parser bug so far was found that way.
- Gross minus budget isn't real profit, because marketing isn't in the budget and cinemas keep part of the gross.
- A film's cast centrality should use only films released before it, or success leaks into the predictor. The same goes for awards.
- Wikidata's series property also holds lists such as "100 greatest films", so it isn't a clean franchise flag.
- See "What's still open" in [docs/imdb-data-structure.md](docs/imdb-data-structure.md) for the current problems.
