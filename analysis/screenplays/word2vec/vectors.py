"""Script vectors per film and layer (DECISIONS.md, "Test settings": script vector, TF-IDF representation).

For each trained space private/w2v_w{W}_s{S}.kv and each layer:
  tfidf_mean  sum_w tf(w) idf(w) v_w / sum_w tf(w) idf(w)    (primary)
  plain_mean  sum_w tf(w) v_w / sum_w tf(w)                  (robustness)
tf = count of the word in that layer of that film, idf = ln(N / df) over the N = 2188 documents of that layer.
Words outside the space's vocabulary (min_count 20) are dropped.

Also writes the TF-IDF representation itself (tf x idf, rows L2-normalised) over the same vocabulary.

Writes private/vectors_w{W}_s{S}.npz and private/tfidf_{layer}.npz (gitignored).
Run:  python -I scripts-analysis/word2vec/vectors.py
"""
import pickle
from pathlib import Path

import numpy as np
from gensim.models import KeyedVectors
from scipy import sparse

PRIVATE = Path(__file__).resolve().parent.parent / "private"
LAYERS = ("dialogue", "description")


def count_matrix(counts, ids, vocab, layer):
    index = {w: i for i, w in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for r, fid in enumerate(ids):
        for w, c in counts[fid][layer].items():
            j = index.get(w)
            if j is not None:
                rows.append(r)
                cols.append(j)
                vals.append(c)
    return sparse.csr_matrix((np.array(vals, float), (rows, cols)), shape=(len(ids), len(vocab)))


def idf(C):
    df = np.asarray((C > 0).sum(axis=0)).ravel()
    with np.errstate(divide="ignore"):
        out = np.log(C.shape[0] / df)
    out[df == 0] = 0.0
    return out


def main():
    with open(PRIVATE / "counts.pkl", "rb") as f:
        counts = pickle.load(f)
    ids = sorted(counts)
    models = sorted(PRIVATE.glob("w2v_w*_s*.kv"))
    if not models:
        raise SystemExit("no trained spaces in private/")
    vocab0 = None
    for path in models:
        kv = KeyedVectors.load(str(path))
        vocab = list(kv.index_to_key)
        if vocab0 is None:
            vocab0 = vocab
            mats = {k: count_matrix(counts, ids, vocab, k) for k in LAYERS}
            idfs = {k: idf(mats[k]) for k in LAYERS}
            for k in LAYERS:
                T = sparse.csr_matrix(mats[k].multiply(idfs[k][None, :]))
                norms = np.sqrt(np.asarray(T.multiply(T).sum(axis=1)).ravel())
                T = sparse.diags(1 / np.where(norms > 0, norms, 1)) @ T
                sparse.save_npz(PRIVATE / f"tfidf_{k}.npz", sparse.csr_matrix(T))
                print(f"{k}: vocab {len(vocab)}, words with idf 0: {(idfs[k] == 0).sum()}, "
                      f"films with an empty row: {(norms == 0).sum()}")
        elif sorted(vocab) != sorted(vocab0):
            raise SystemExit(f"{path.name}: vocabulary differs from {models[0].name}")
        order = [kv.key_to_index[w] for w in vocab0]
        V = kv.vectors[order]
        out = {"ids": np.array(ids)}
        for k in LAYERS:
            C = mats[k]
            W = sparse.csr_matrix(C.multiply(idfs[k][None, :]))
            for name, M in (("tfidf_mean", W), ("plain_mean", C)):
                s = np.asarray(M.sum(axis=1)).ravel()
                out[f"{k}_{name}"] = np.asarray(M @ V) / np.where(s > 0, s, np.nan)[:, None]
        np.savez(PRIVATE / f"vectors_{path.stem[4:]}.npz", **out)
        print(f"{path.name}: written")
    np.save(PRIVATE / "vector_ids.npy", np.array(ids))


if __name__ == "__main__":
    main()
