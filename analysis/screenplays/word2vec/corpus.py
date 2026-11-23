"""Build the Word2Vec training corpus and the per-film word counts (DECISIONS.md, "Test settings").

Uses preprocess.film_layers with the same common-label filter as preprocess.main(): a label token in
NAME_MAX_FILMS or more of the 2188 scripts is a common word and is not removed as a name.

Writes, in scripts-analysis/private/ (gitignored, it holds script text):
  corpus.txt   one element per line (both layers, films in imdb id order, dialogue then description), tokens
               separated by spaces. A Word2Vec window never crosses a line.
  counts.pkl   {imdb_id: {"dialogue": Counter, "description": Counter}} of the kept tokens.
Writes corpus_summary.csv (numbers only) next to this file.

Run:  IMDB_DATA_DIR=... python -I scripts-analysis/word2vec/corpus.py
"""
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocess"))
sys.path.insert(0, str(HERE.parent / "data_check"))
sys.path.insert(0, str(HERE.parent / "genre_counts"))
from genre_counts import DATA  # noqa: E402
from parse_layers import one_per_id, parse_script, read_records  # noqa: E402
from preprocess import LAYERS, NAME_MAX_FILMS, film_layers, label_tokens  # noqa: E402

PRIVATE = HERE.parent / "private"


def main():
    records = sorted(one_per_id(read_records(DATA)), key=lambda r: r["imdb_id"])
    parsed = [(r["imdb_id"], parse_script(r["script"])) for r in records]
    label_df = Counter()
    for _, els in parsed:
        label_df.update(label_tokens(els))
    common = {t for t, c in label_df.items() if c >= NAME_MAX_FILMS}
    PRIVATE.mkdir(exist_ok=True)
    counts, lines, tokens = {}, Counter(), Counter()
    with open(PRIVATE / "corpus.txt", "w") as f:
        for fid, els in parsed:
            layers, _, _ = film_layers(els, common)
            counts[fid] = {}
            for k in LAYERS:
                c = Counter()
                for el in layers[k]:
                    f.write(" ".join(el) + "\n")
                    c.update(el)
                counts[fid][k] = c
                lines[k] += len(layers[k])
                tokens[k] += sum(c.values())
    with open(PRIVATE / "counts.pkl", "wb") as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)
    rows = [("films", len(records)), ("common_label_tokens", len(common))]
    rows += [(f"{k}_{m}", v[k]) for k in LAYERS for m, v in (("elements", lines), ("tokens", tokens))]
    with open(HERE / "corpus_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "value"])
        w.writerows(rows)
    for r in rows:
        print(*r)


if __name__ == "__main__":
    main()
