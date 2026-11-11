"""Shared machinery: document-term matrices, Scattertext rates and scores, and the shuffle null.

One script = one document. A comparison is two groups of documents, "low" (unsuccessful) and "high"
(successful). Everything here works on counts; no text leaves private/.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

from . import s03_clean
from .config import cfg
from .paths import PRIVATE

DTM_DIR = PRIVATE / "dtm"
DTM_DIR.mkdir(parents=True, exist_ok=True)


def variant_key(layer: str, names: str, stopwords: str, lemmatize: str) -> str:
    return f"{layer}-names_{names}-stop_{stopwords}-lemma_{lemmatize}"


def dtm(layer: str = "dialogue", names: str = "labels", stopwords: str = "kept", lemmatize: str = "off"):
    """(X, vocab, doc_ids, doc_lengths) for every script; cached. doc_lengths are all tokens kept by the
    variant, so rates are per 10,000 of the words the variant counts."""
    key = variant_key(layer, names, stopwords, lemmatize)
    f = DTM_DIR / f"{key}.npz"
    meta = DTM_DIR / f"{key}.json"
    if f.exists() and meta.exists():
        m = json.loads(meta.read_text())
        return sp.load_npz(f).tocsr(), np.array(m["vocab"]), np.array(m["ids"]), np.array(m["lengths"])
    tok = s03_clean.load_tokens()
    t = tok[tok.layer == layer].sort_values("tconst")
    streams = [s03_clean.stream(r, names=names, stopwords=stopwords, lemmatize=lemmatize) for _, r in t.iterrows()]
    cv = CountVectorizer(analyzer=lambda x: x, lowercase=False)
    X = cv.fit_transform(streams).tocsr().astype(np.int32)
    vocab = cv.get_feature_names_out()
    lengths = np.array([len(s) for s in streams])
    sp.save_npz(f, X)
    meta.write_text(json.dumps({"vocab": vocab.tolist(), "ids": t.tconst.tolist(), "lengths": lengths.tolist()}))
    return X, vocab, t.tconst.to_numpy(), lengths


def excluded_scripts() -> set[str]:
    """Scripts whose dialogue markup failed (fewer than text.min_dialogue_words tagged dialogue words).
    They are left out of every comparison and counted in Step 2."""
    from . import s02_layers
    rep = s02_layers.load()["report"]
    return set(rep.tconst[rep.dialogue_words < cfg()["text"]["min_dialogue_words"]])


def comparison(X, ids, lengths, labels: pd.Series):
    """Restrict to the documents labelled low or high, minus the excluded scripts. labels is indexed
    by tconst."""
    lab = labels.reindex(ids)
    keep = (lab.isin(["low", "high"]) & ~pd.Series(ids).isin(excluded_scripts()).to_numpy()).to_numpy()
    y = (lab[keep] == "high").to_numpy().astype(np.int8)
    return X[keep], ids[keep], lengths[keep], y


def scores(X, lengths, y, min_df: int, mode: str = "repeated"):
    """Rates per group and the log10 ratio score, for terms used by at least min_df documents.
    mode 'repeated': rate per 10,000 words; 'presence': share of the group's scripts using the term."""
    c = cfg()["scattertext"]
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    ok = df >= min_df
    if mode == "repeated":
        hi_counts = np.asarray(X[y == 1].sum(axis=0)).ravel()
        lo_counts = np.asarray(X[y == 0].sum(axis=0)).ravel()
        r_hi = c["rate_per"] * hi_counts / lengths[y == 1].sum()
        r_lo = c["rate_per"] * lo_counts / lengths[y == 0].sum()
        e = c["smoothing_per_10k"]
    else:
        B = (X > 0).astype(np.int8)
        r_hi = np.asarray(B[y == 1].sum(axis=0)).ravel() / max(1, (y == 1).sum())
        r_lo = np.asarray(B[y == 0].sum(axis=0)).ravel() / max(1, (y == 0).sum())
        e = c["smoothing_presence"]
    score = np.log10((r_hi + e) / (r_lo + e))
    return {"ok": ok, "df": df, "rate_high": r_hi, "rate_low": r_lo, "score": score}


def term_table(vocab, res, X, y) -> pd.DataFrame:
    ok = res["ok"]
    Xb = (X > 0)
    t = pd.DataFrame({"term": vocab[ok], "doc_freq": res["df"][ok],
                      "docs_high": np.asarray(Xb[y == 1].sum(axis=0)).ravel()[ok],
                      "docs_low": np.asarray(Xb[y == 0].sum(axis=0)).ravel()[ok],
                      "rate_high": res["rate_high"][ok], "rate_low": res["rate_low"][ok],
                      "score": res["score"][ok]})
    return t.sort_values("score", ascending=False).reset_index(drop=True)


def top_terms(table: pd.DataFrame, n: int) -> tuple[set, set]:
    hi = set(table.nlargest(n, "score").term)
    lo = set(table.nsmallest(n, "score").term)
    return hi, lo


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def stratified_permutation(y, strata, rng):
    out = y.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def shuffle_null(X, lengths, y, strata, min_df: int, shuffles: int, seed: int, mode: str = "repeated",
                 progress: bool = True):
    """Recompute every term's score under label shuffles (within strata if given).

    Returns the per-term null mean and standard deviation, how often each term's null score is at
    least as extreme as its real one, the null distribution of the number of terms above the
    threshold, and the null distribution of the largest |score| (for a family-wise test)."""
    thr = cfg()["scattertext"]["threshold"]
    real = scores(X, lengths, y, min_df, mode)
    ok = real["ok"]
    rs = real["score"][ok]
    rng = np.random.default_rng(seed)
    s1 = np.zeros(ok.sum())
    s2 = np.zeros(ok.sum())
    extreme = np.zeros(ok.sum(), dtype=np.int64)
    n_above, n_above_hi, n_above_lo, max_abs = [], [], [], []
    for i in range(shuffles):
        yp = stratified_permutation(y, strata, rng) if strata is not None else rng.permutation(y)
        ns = scores(X, lengths, yp, min_df, mode)["score"][ok]
        s1 += ns
        s2 += ns * ns
        extreme += np.abs(ns) >= np.abs(rs)
        n_above.append(int((np.abs(ns) >= thr).sum()))
        n_above_hi.append(int((ns >= thr).sum()))
        n_above_lo.append(int((ns <= -thr).sum()))
        max_abs.append(float(np.abs(ns).max()))
        if progress and (i + 1) % max(1, shuffles // 10) == 0:
            print(f"shuffles: {i + 1}/{shuffles}", flush=True)
    mean = s1 / shuffles
    std = np.sqrt(np.maximum(s2 / shuffles - mean ** 2, 0))
    return {"real": real, "mean": mean, "std": std, "extreme": extreme, "n_above": np.array(n_above),
            "n_above_hi": np.array(n_above_hi), "n_above_lo": np.array(n_above_lo), "max_abs": np.array(max_abs),
            "shuffles": shuffles}


def empirical_p(count_as_extreme, shuffles):
    """(1 + #shuffles at least as extreme) / (1 + #shuffles): never exactly 0."""
    return (1 + np.asarray(count_as_extreme)) / (1 + shuffles)


def bh(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def fingerprint(*arrays) -> str:
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:10]
