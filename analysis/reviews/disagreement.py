"""Money–audience disagreement via rank residuals.

Measures: residual of rank(IMDb averageRating) regressed on
rank(log(gross/budget)). Positive = liked more than money rank predicts;
negative = liked less.
Misses: ranks are within the chosen sample only; residual is linear in rank
space, not dollars; extreme profit quintiles are a separate cut for display.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .paths import OUTPUTS


def add_disagreement_residual(
    df: pd.DataFrame,
    profit_col: str = "log_gross_over_budget",
    rating_col: str = "imdb_rating",
) -> pd.DataFrame:
    """Add rank(IMDb) ~ rank(profit) OLS residual and profit quintile."""
    out = df.copy()
    out["rank_log_gross_over_budget"] = np.nan
    out["rank_imdb_rating"] = np.nan
    out["disagreement_residual"] = np.nan
    out["profit_quintile"] = np.nan
    out["disagreement_group"] = pd.NA

    mask = out[profit_col].notna() & out[rating_col].notna()
    if mask.sum() < 5:
        return out

    sub = out.loc[mask].copy()
    # Average ranks for ties (scipy default).
    r_x = stats.rankdata(sub[profit_col].to_numpy(), method="average")
    r_y = stats.rankdata(sub[rating_col].to_numpy(), method="average")
    # OLS through origin-free intercept: y_hat = a + b x
    slope, intercept, *_ = stats.linregress(r_x, r_y)
    resid = r_y - (intercept + slope * r_x)

    out.loc[mask, "rank_log_gross_over_budget"] = r_x
    out.loc[mask, "rank_imdb_rating"] = r_y
    out.loc[mask, "disagreement_residual"] = resid

    # Profit quintiles (1 = lowest profit proxy, 5 = highest).
    try:
        q = pd.qcut(sub[profit_col], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        out.loc[mask, "profit_quintile"] = pd.to_numeric(q, errors="coerce")
    except ValueError:
        out.loc[mask, "profit_quintile"] = np.nan

    # Disagreement groups = extreme residual quintiles.
    try:
        rq = pd.qcut(
            pd.Series(resid, index=sub.index),
            5,
            labels=[1, 2, 3, 4, 5],
            duplicates="drop",
        )
        rq_num = pd.to_numeric(rq, errors="coerce")
        group = pd.Series("middle", index=sub.index, dtype="object")
        group.loc[rq_num == 5] = "rating_above_money"  # liked beyond money rank
        group.loc[rq_num == 1] = "rating_below_money"  # liked less than money rank
        out.loc[mask, "disagreement_group"] = group
    except ValueError:
        pass

    return out


def export_top_disagreements(df: pd.DataFrame, top_n: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-n positive and negative disagreement residuals."""
    d = df.dropna(subset=["disagreement_residual"]).copy()
    keep = [
        "imdb_id",
        "title",
        "year",
        "genres",
        "log_gross_over_budget",
        "gross_minus_budget",
        "imdb_rating",
        "metacritic",
        "rotten_tomatoes",
        "rank_log_gross_over_budget",
        "rank_imdb_rating",
        "disagreement_residual",
        "profit_quintile",
        "disagreement_group",
    ]
    keep = [c for c in keep if c in d.columns]
    above = d.nlargest(top_n, "disagreement_residual")[keep]
    below = d.nsmallest(top_n, "disagreement_residual")[keep]
    above.to_csv(OUTPUTS / "disagreement_top_rating_above_money.csv", index=False)
    below.to_csv(OUTPUTS / "disagreement_top_rating_below_money.csv", index=False)
    return above, below


def disagreement_summary(df: pd.DataFrame) -> pd.Series:
    d = df.dropna(subset=["disagreement_residual"])
    profit = df.dropna(subset=["log_gross_over_budget"])
    q1 = profit[profit["profit_quintile"] == 1]
    q5 = profit[profit["profit_quintile"] == 5]
    summary = {
        "n_with_residual": int(len(d)),
        "residual_mean": float(d["disagreement_residual"].mean()),
        "residual_std": float(d["disagreement_residual"].std(ddof=1)),
        "n_rating_above_money": int((d["disagreement_group"] == "rating_above_money").sum()),
        "n_rating_below_money": int((d["disagreement_group"] == "rating_below_money").sum()),
        "n_middle": int((d["disagreement_group"] == "middle").sum()),
        "n_profit_q1": int(len(q1)),
        "n_profit_q5": int(len(q5)),
        "profit_q1_max": float(q1["log_gross_over_budget"].max()) if len(q1) else np.nan,
        "profit_q5_min": float(q5["log_gross_over_budget"].min()) if len(q5) else np.nan,
        "mean_imdb_profit_q1": float(q1["imdb_rating"].mean()) if len(q1) else np.nan,
        "mean_imdb_profit_q5": float(q5["imdb_rating"].mean()) if len(q5) else np.nan,
    }
    pd.Series(summary).to_csv(OUTPUTS / "disagreement_summary.csv", header=["value"])
    return pd.Series(summary)
