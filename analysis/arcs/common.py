"""Helpers shared by several steps: genre and decade bins, pooled cells, residualisation, r_min."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import cfg
from .paths import OUTPUTS

INSTRUMENT_NAMES = {"vader": "VADER", "sst2": "DistilBERT SST-2", "roberta3": "Twitter-RoBERTa (3-class)",
                    "modernbert3": "ModernBERT-large (3-class)"}


def primary_instrument() -> str:
    """The instrument the fallback rule chose (Step 3), or VADER until the rule has been applied."""
    p = OUTPUTS / "s03_instrument_decision.json"
    if p.exists():
        import json
        return json.loads(p.read_text())["primary_instrument"]
    return "vader"


def primary_genre(genres: str) -> str:
    g = set((genres or "").split(","))
    for name in cfg()["controls"]["primary_genre_priority"]:
        if name in g:
            return name
    return cfg()["controls"]["other_genre"]


def decade_bin(year: int) -> str:
    for lo, hi, label in cfg()["controls"]["decade_bins"]:
        if lo <= year <= hi:
            return label
    return "other"


def pooled_cells(genre: pd.Series, decade: pd.Series, min_n: int | None = None) -> pd.Series:
    """Genre x decade cells for the films given. A cell with fewer than min_n films is pooled into
    '<genre> | pooled decades'; if that is still too small, into 'pooled'. Computed on the sample analysed,
    so a singleton cell never gives a zero residual or a no-op permutation."""
    min_n = min_n or cfg()["controls"]["min_films_per_cell"]
    cell = genre + " | " + decade
    n = cell.map(cell.value_counts())
    cell = cell.where(n >= min_n, genre + " | pooled decades")
    n = cell.map(cell.value_counts())
    return cell.where(n >= min_n, "pooled")


def design(cells: pd.Series, covariates: pd.DataFrame | None) -> np.ndarray:
    """Intercept-free one-hot of cells (one column per cell) plus centred covariates."""
    X = pd.get_dummies(cells.astype(str)).to_numpy(float)
    if covariates is not None and covariates.shape[1]:
        C = covariates.to_numpy(float)
        X = np.column_stack([X, C - C.mean(0)])
    return X


def residualise(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residual of y on X (Frisch-Waugh: the part of y the controls don't explain)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def r_min(n: int, alpha: float, power: float = 0.80) -> float:
    """Smallest |r| detectable with the given power at two-sided alpha (Fisher z approximation)."""
    k = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(np.tanh(k / np.sqrt(n - 3)))


def empirical_p(n_as_extreme: int, n: int) -> float:
    return (1 + int(n_as_extreme)) / (1 + int(n))
