"""Label-shuffle nulls for claimed associations.

Primary null: shuffle the response **within genre × decade strata** (preserves
genre/decade margins). Secondary: a global (unstratified) shuffle for comparison.

Measures: how often a Spearman ρ as extreme as the observed one appears under
the stratified shuffle; z = (ρ − null_mean) / null_std.
Misses: multi-genre films use the first-listed genre only; size-1 strata cannot
be shuffled; does not preserve network structure.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats


def primary_genre(genres: pd.Series) -> pd.Series:
    """First-listed IMDb genre (comma-separated)."""
    return genres.fillna("").astype(str).str.split(",").str[0].str.strip().replace("", "Unknown")


def decade_label(year: pd.Series) -> pd.Series:
    y = pd.to_numeric(year, errors="coerce")
    return (y // 10 * 10).astype("Int64").astype(str).replace("<NA>", "Unknown")


def genre_decade_strata(df: pd.DataFrame, genre_col: str = "genres", year_col: str = "year") -> pd.Series:
    return primary_genre(df[genre_col]) + "|" + decade_label(df[year_col])


def _shuffle_within_strata(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for key in np.unique(strata):
        idx = np.where(strata == key)[0]
        if len(idx) < 2:
            continue
        out[idx] = rng.permutation(out[idx])
    return out


def spearman_with_null(
    x: pd.Series,
    y: pd.Series,
    n_perm: int = 1000,
    random_state: int = 0,
    strata: pd.Series | None = None,
) -> dict:
    """Observed Spearman ρ vs stratified shuffle (primary) and global shuffle (secondary).

    If ``strata`` is None, only the global null is computed (stratified fields NaN).
    """
    frame = pd.DataFrame({"x": x, "y": y})
    if strata is not None:
        frame["strata"] = strata.values if hasattr(strata, "values") else strata
    df = frame.dropna(subset=["x", "y"] + (["strata"] if strata is not None else []))
    if len(df) < 3:
        return {
            "n": int(len(df)),
            "rho": np.nan,
            "p_analytic": np.nan,
            "null_mean_strat": np.nan,
            "null_std_strat": np.nan,
            "z_strat": np.nan,
            "p_null_strat": np.nan,
            "null_mean_global": np.nan,
            "null_std_global": np.nan,
            "p_null_global": np.nan,
        }

    rho, p_analytic = stats.spearmanr(df["x"], df["y"])
    rng = np.random.default_rng(random_state)
    y_vals = df["y"].to_numpy()
    x_vals = df["x"].to_numpy()

    # Global (secondary)
    null_g = np.empty(n_perm)
    for i in range(n_perm):
        null_g[i], _ = stats.spearmanr(x_vals, rng.permutation(y_vals))
    p_global = (np.sum(np.abs(null_g) >= abs(rho)) + 1) / (n_perm + 1)

    out = {
        "n": int(len(df)),
        "rho": float(rho),
        "p_analytic": float(p_analytic),
        "null_mean_global": float(null_g.mean()),
        "null_std_global": float(null_g.std(ddof=1)),
        "p_null_global": float(p_global),
    }

    if strata is None or "strata" not in df.columns:
        out.update(
            {
                "null_mean_strat": np.nan,
                "null_std_strat": np.nan,
                "z_strat": np.nan,
                "p_null_strat": np.nan,
            }
        )
        return out

    strata_vals = df["strata"].astype(str).to_numpy()
    null_s = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = _shuffle_within_strata(y_vals, strata_vals, rng)
        null_s[i], _ = stats.spearmanr(x_vals, shuffled)
    mean_s = float(null_s.mean())
    std_s = float(null_s.std(ddof=1))
    z = float((rho - mean_s) / std_s) if std_s > 0 else np.nan
    p_strat = (np.sum(np.abs(null_s) >= abs(rho)) + 1) / (n_perm + 1)
    out.update(
        {
            "null_mean_strat": mean_s,
            "null_std_strat": std_s,
            "z_strat": z,
            "p_null_strat": float(p_strat),
        }
    )
    return out


def statistic_with_null(
    values: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    shuffle_col: str,
    n_perm: int = 1000,
    random_state: int = 0,
    strata_col: str | None = None,
) -> dict:
    """Generic observed statistic vs stratified (primary) and global (secondary) shuffle."""
    need = [shuffle_col] + ([strata_col] if strata_col else [])
    df = values.dropna(subset=need).copy()
    observed = float(stat_fn(df))
    rng = np.random.default_rng(random_state)
    col = df[shuffle_col].to_numpy()

    null_g = np.empty(n_perm)
    for i in range(n_perm):
        trial = df.copy()
        trial[shuffle_col] = rng.permutation(col)
        null_g[i] = float(stat_fn(trial))

    result = {
        "n": int(len(df)),
        "observed": observed,
        "null_mean_global": float(null_g.mean()),
        "null_std_global": float(null_g.std(ddof=1)),
        "p_null_global": float((np.sum(np.abs(null_g) >= abs(observed)) + 1) / (n_perm + 1)),
    }

    if not strata_col:
        result.update({"null_mean_strat": np.nan, "null_std_strat": np.nan, "z_strat": np.nan, "p_null_strat": np.nan})
        return result

    strata = df[strata_col].astype(str).to_numpy()
    null_s = np.empty(n_perm)
    for i in range(n_perm):
        trial = df.copy()
        trial[shuffle_col] = _shuffle_within_strata(col, strata, rng)
        null_s[i] = float(stat_fn(trial))
    mean_s = float(null_s.mean())
    std_s = float(null_s.std(ddof=1))
    result.update(
        {
            "null_mean_strat": mean_s,
            "null_std_strat": std_s,
            "z_strat": float((observed - mean_s) / std_s) if std_s > 0 else np.nan,
            "p_null_strat": float((np.sum(np.abs(null_s) >= abs(observed)) + 1) / (n_perm + 1)),
        }
    )
    return result


def correlation_table(
    df: pd.DataFrame,
    columns: list[str],
    n_perm: int = 1000,
    random_state: int = 0,
    strata: pd.Series | None = None,
) -> pd.DataFrame:
    """Pairwise Spearman ρ with stratified null (primary) and global null (secondary)."""
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            # Align strata to pairwise-complete rows inside spearman_with_null via frame join.
            if strata is not None:
                sub = pd.DataFrame({"x": df[a], "y": df[b], "strata": strata}).dropna()
                res = spearman_with_null(sub["x"], sub["y"], n_perm=n_perm, random_state=random_state, strata=sub["strata"])
            else:
                res = spearman_with_null(df[a], df[b], n_perm=n_perm, random_state=random_state)
            rows.append({"a": a, "b": b, **res})
    return pd.DataFrame(rows)
