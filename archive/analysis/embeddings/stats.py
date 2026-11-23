"""The measurements of Step 3, as small functions tested on planted signals (tests/test_embeddings_clean.py).

All of them ask the same thing in different ways: do films whose text sits close together in the vector
space have similar money outcomes?
"""
from __future__ import annotations

import os

import numpy as np
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def _fit_predict(Z, y, ctrl, tr, te, alphas):
    """Controls by least squares, then a ridge regression of what the controls leave on the vectors.
    Returns the two predictions for the held-out films."""
    beta, *_ = np.linalg.lstsq(ctrl[tr], y[tr], rcond=None)
    sc = StandardScaler().fit(Z[tr])
    ridge = RidgeCV(alphas=alphas).fit(sc.transform(Z[tr]), y[tr] - ctrl[tr] @ beta)
    return ctrl[te] @ beta, ridge.predict(sc.transform(Z[te]))


def oof(Z, y, ctrl, alphas, folds: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions: every film is predicted by a model that never saw it."""
    base, emb = np.empty(len(y)), np.empty(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(Z):
        base[te], emb[te] = _fit_predict(Z, y, ctrl, tr, te, alphas)
    return base, emb


def summarise(y, base, emb) -> dict:
    left = y - base
    return {"r2_base": r2(y, base), "r2_full": r2(y, base + emb), "r2_gain": r2(y, base + emb) - r2(y, base),
            "partial_pearson": float(stats.pearsonr(left, emb)[0]),
            "partial_spearman": float(stats.spearmanr(left, emb)[0])}


def cv_scores(Z, y, ctrl, alphas, folds: int, seed: int) -> dict:
    return summarise(y, *oof(Z, y, ctrl, alphas, folds, seed))


def permute_within(strata: np.ndarray, rng) -> np.ndarray:
    """A permutation of the rows that only swaps films inside the same stratum."""
    idx = np.arange(len(strata))
    for s in np.unique(strata):
        m = np.flatnonzero(strata == s)
        idx[m] = rng.permutation(m)
    return idx


def n_jobs() -> int:
    """Worker processes for the shuffles: EMBED_JOBS, else LSF's core count, else 1."""
    return int(os.environ.get("EMBED_JOBS") or os.environ.get("LSB_DJOB_NUMPROC") or 1)


def _gain(Z, idx, y, ctrl, alphas, folds, seed):
    return cv_scores(Z[idx], y, ctrl, alphas, folds, seed)["r2_gain"]


def shuffle_gains(Z, y, ctrl, strata, alphas, folds: int, perms: int, seed: int) -> list[float]:
    """The null: each film gets the vector of another film from its own stratum, which breaks the link
    between text and money and keeps everything else (the vectors, the outcomes, the controls). The
    permutations are drawn first, in order, so the result doesn't depend on how many workers run them."""
    rng = np.random.default_rng(seed)
    idx = [permute_within(strata, rng) for _ in range(perms)]
    jobs = n_jobs()
    if jobs <= 1:
        return [_gain(Z, i, y, ctrl, alphas, folds, seed) for i in idx]
    from joblib import Parallel, delayed, parallel_config
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=jobs)(delayed(_gain)(Z, i, y, ctrl, alphas, folds, seed) for i in idx)


def bootstrap_gain(y, base, emb, n: int, seed: int) -> tuple[float, float]:
    """95% interval of the R2 gain, resampling films."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        out.append(r2(y[i], base[i] + emb[i]) - r2(y[i], base[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def time_split(Z, y, ctrl, year, train_share: float, alphas) -> dict:
    """Fit on the earliest films, score on the later ones: the only version that could be used in advance."""
    cut = int(np.quantile(year, train_share))
    tr, te = np.flatnonzero(year <= cut), np.flatnonzero(year > cut)
    base, emb = _fit_predict(Z, y, ctrl, tr, te, alphas)
    return {"cutoff_year": cut, "n_train": len(tr), "n_test": len(te), **summarise(y[te], base, emb)}


def knn_edges(V: np.ndarray, k: int) -> np.ndarray:
    """The similarity network: each film linked to its k nearest films by cosine. Undirected, no duplicates."""
    U = V / np.linalg.norm(V, axis=1, keepdims=True)
    sim = U @ U.T
    np.fill_diagonal(sim, -np.inf)
    nb = np.argpartition(-sim, k, axis=1)[:, :k]
    e = np.column_stack([np.repeat(np.arange(len(U)), k), nb.ravel()])
    return np.unique(np.sort(e, axis=1), axis=0)


def assortativity(edges: np.ndarray, a: np.ndarray) -> float:
    """Newman's assortativity coefficient for a number on the nodes: the Pearson correlation of the
    values at the two ends of an edge, each edge counted in both directions."""
    x = np.concatenate([a[edges[:, 0]], a[edges[:, 1]]])
    z = np.concatenate([a[edges[:, 1]], a[edges[:, 0]]])
    return float(np.corrcoef(x, z)[0, 1])


def assortativity_null(edges, a, strata, perms: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [assortativity(edges, a[permute_within(strata, rng)]) for _ in range(perms)]


def p_upper(obs: float, null) -> float:
    return float((1 + sum(n >= obs for n in null)) / (1 + len(null)))
