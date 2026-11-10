"""Label-shuffle nulls for claimed associations.

Measures: how often a statistic as extreme as the observed one appears when
labels are randomly reassigned (breaking the pairing while keeping margins).
Misses: does not preserve network structure or within-genre dependence; use a
degree-preserving null if the claim is about a graph.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def spearman_with_null(
    x: pd.Series,
    y: pd.Series,
    n_perm: int = 1000,
    random_state: int = 0,
) -> dict:
    """Observed Spearman rho vs label-shuffle null (shuffle y)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3:
        return {"n": len(df), "rho": np.nan, "p_analytic": np.nan, "p_null": np.nan, "null_mean": np.nan}
    rho, p_analytic = stats.spearmanr(df["x"], df["y"])
    rng = np.random.default_rng(random_state)
    y_vals = df["y"].to_numpy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = rng.permutation(y_vals)
        null[i], _ = stats.spearmanr(df["x"], shuffled)
    p_null = (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)
    return {
        "n": int(len(df)),
        "rho": float(rho),
        "p_analytic": float(p_analytic),
        "p_null": float(p_null),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
    }


def statistic_with_null(
    values: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    shuffle_col: str,
    n_perm: int = 1000,
    random_state: int = 0,
) -> dict:
    """Generic observed statistic vs shuffling one column."""
    df = values.dropna().copy()
    observed = float(stat_fn(df))
    rng = np.random.default_rng(random_state)
    null = np.empty(n_perm)
    col = df[shuffle_col].to_numpy()
    for i in range(n_perm):
        trial = df.copy()
        trial[shuffle_col] = rng.permutation(col)
        null[i] = float(stat_fn(trial))
    p_null = (np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1)
    return {
        "n": int(len(df)),
        "observed": observed,
        "p_null": float(p_null),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
    }


def correlation_table(
    df: pd.DataFrame,
    columns: list[str],
    n_perm: int = 1000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Pairwise Spearman rhos with label-shuffle null p-values."""
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            res = spearman_with_null(df[a], df[b], n_perm=n_perm, random_state=random_state)
            rows.append({"a": a, "b": b, **res})
    return pd.DataFrame(rows)
