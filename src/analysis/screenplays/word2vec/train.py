"""Train one Word2Vec space (DECISIONS.md: skip-gram, negative sampling, 100 dimensions, min_count 20, workers=1).

Run:  python -I analysis/screenplays/word2vec/train.py --window 5 --seed 1
Writes private/w2v_w{window}_s{seed}.kv (gitignored). --limit N trains on the first N lines only (for the
reproducibility check) and writes private/check_w{window}_s{seed}_{tag}.kv.
"""
import argparse
import itertools
import logging
import time
from pathlib import Path

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

PRIVATE = Path(__file__).resolve().parent.parent / "private"
DIM, MIN_COUNT = 100, 20


class Limited:
    def __init__(self, path, limit):
        self.path, self.limit = path, limit

    def __iter__(self):
        return itertools.islice(LineSentence(str(self.path)), self.limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tag", default="a")
    a = ap.parse_args()
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    corpus = PRIVATE / "corpus.txt"
    sents = Limited(corpus, a.limit) if a.limit else LineSentence(str(corpus))
    t = time.time()
    m = Word2Vec(sentences=sents, vector_size=DIM, window=a.window, min_count=MIN_COUNT, sg=1, hs=0,
                 negative=5, workers=1, seed=a.seed)
    name = f"check_w{a.window}_s{a.seed}_{a.tag}.kv" if a.limit else f"w2v_w{a.window}_s{a.seed}.kv"
    m.wv.save(str(PRIVATE / name))
    print(f"done {name}: vocab {len(m.wv)}, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
