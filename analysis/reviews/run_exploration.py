"""Compute review-analysis outputs used by 01_exploration.ipynb.

Writes only derived numbers / figure files under outputs/. Review text stays in
build/imdb/features/reviews_private/ (gitignored via build/).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.reviews.build_movie_table import build_movie_table, primary_sample, save_movie_table
from analysis.reviews.nulls import correlation_table, spearman_with_null
from analysis.reviews.paths import HAND_LABEL_CSV, OUTPUTS, PRIVATE, TEXT_FEATURES
from analysis.reviews.plotting import apply_bw_style, annotate_extremes, save_fig, scatter_bw
from analysis.reviews.scattertext_compare import run_comparison, top_term_examples
from analysis.reviews.text_features import (
    agreement_from_hand_labels,
    compute_movie_text_features,
    export_hand_label_sample,
    genre_sentiment_table,
    sentiment_profit_by_genre_null,
    validate_vader_on_stanford,
)

MAX_YEAR = 2017


def load_or_build_table() -> pd.DataFrame:
    table = build_movie_table(max_year=MAX_YEAR)
    save_movie_table(table)
    return table


def idea1(primary: pd.DataFrame) -> dict:
    """Spearman money/critics/audience + quadrants + scatters."""
    apply_bw_style()
    cols = {
        "log_gross_over_budget": "log(gross/budget)",
        "imdb_rating": "IMDb averageRating",
        "metacritic": "Metacritic",
        "rotten_tomatoes": "Rotten Tomatoes %",
    }
    sub = primary[list(cols)].dropna(how="any")
    # Also report pairwise with pairwise-complete data.
    pair_cols = list(cols)
    corr = correlation_table(primary, pair_cols, n_perm=1000, random_state=0)
    corr.to_csv(OUTPUTS / "spearman_correlations.csv", index=False)

    # Scatter grid
    pairs = [
        ("log_gross_over_budget", "imdb_rating"),
        ("log_gross_over_budget", "metacritic"),
        ("log_gross_over_budget", "rotten_tomatoes"),
        ("imdb_rating", "metacritic"),
        ("imdb_rating", "rotten_tomatoes"),
        ("metacritic", "rotten_tomatoes"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (a, b) in zip(axes.ravel(), pairs):
        d = primary[[a, b]].dropna()
        scatter_bw(ax, d[a], d[b], cols[a], cols[b], alpha=0.25)
        # Break-even / mid-score guides where meaningful.
        if a == "log_gross_over_budget":
            ax.axvline(0.0, color="black", lw=0.8, ls="--")
    caption = (
        "Figure. Pairwise relationships among log(gross/budget), IMDb rating, "
        "Metacritic and Rotten Tomatoes for the 100k+ clean theatrical sample "
        f"with year ≤ {MAX_YEAR}. Dashed line marks break-even (gross = budget). "
        "Points are individual movies; axes are labelled in native units."
    )
    save_fig(fig, OUTPUTS / "fig_scatter_money_scores.png", caption)

    # Quadrants: profitable but poorly rated / loved but unprofitable
    d = primary.dropna(subset=["log_gross_over_budget", "imdb_rating"]).copy()
    # Profit threshold: break-even; rating threshold: sample median.
    x_thr, y_thr = 0.0, float(d["imdb_rating"].median())
    profitable_bad = d[(d["log_gross_over_budget"] > x_thr) & (d["imdb_rating"] < y_thr)].copy()
    loved_unprof = d[(d["log_gross_over_budget"] < x_thr) & (d["imdb_rating"] >= y_thr)].copy()
    profitable_bad = profitable_bad.assign(
        disagreement_score=lambda x: x["log_gross_over_budget"] * (y_thr - x["imdb_rating"])
    ).sort_values("disagreement_score", ascending=False)
    loved_unprof = loved_unprof.assign(
        disagreement_score=lambda x: (-x["log_gross_over_budget"]) * (x["imdb_rating"] - y_thr)
    ).sort_values("disagreement_score", ascending=False)

    top_n = 25
    keep_cols = [
        "imdb_id",
        "title",
        "year",
        "genres",
        "log_gross_over_budget",
        "gross_minus_budget",
        "imdb_rating",
        "metacritic",
        "rotten_tomatoes",
        "disagreement_score",
    ]
    profitable_bad.head(top_n)[keep_cols].to_csv(OUTPUTS / "quadrant_profitable_low_imdb.csv", index=False)
    loved_unprof.head(top_n)[keep_cols].to_csv(OUTPUTS / "quadrant_loved_unprofitable.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_bw(
        ax,
        d["log_gross_over_budget"],
        d["imdb_rating"],
        "log(gross/budget)",
        "IMDb averageRating",
        alpha=0.3,
    )
    ax.axvline(x_thr, color="black", lw=0.9, ls="--")
    ax.axhline(y_thr, color="black", lw=0.9, ls="--")
    # Highlight top disagreements
    highlight = pd.concat([profitable_bad.head(8), loved_unprof.head(8)])
    ax.scatter(
        highlight["log_gross_over_budget"],
        highlight["imdb_rating"],
        facecolors="none",
        edgecolors="black",
        s=60,
        linewidths=1.2,
        marker="s",
        label="top disagreements",
    )
    annotate_extremes(ax, highlight, "log_gross_over_budget", "imdb_rating", "title", n=8)
    ax.legend(loc="lower right", fontsize=9)
    q_caption = (
        "Figure. Quadrant view of money vs audience rating (100k+ clean, year ≤ "
        f"{MAX_YEAR}). Vertical dashed line: break-even; horizontal: median IMDb "
        f"rating ({y_thr:.2f}). Squares mark extreme disagreements "
        "(profitable + below-median rating, or unprofitable + above-median rating)."
    )
    save_fig(fig, OUTPUTS / "fig_quadrants_money_imdb.png", q_caption)

    bias = {
        "n_primary": int(len(primary)),
        "n_under_budget": int(primary["under_budget"].sum()),
        "frac_under_budget": float(primary["under_budget"].mean()),
        "imdb_rating_median": y_thr,
        "n_profitable_low_imdb": int(len(profitable_bad)),
        "n_loved_unprofitable": int(len(loved_unprof)),
        "n_complete_score_rows": int(len(sub)),
    }
    pd.Series(bias).to_csv(OUTPUTS / "idea1_summary.csv", header=["value"])
    return {"corr": corr, "bias": bias}


def text_block(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ids = set(primary.loc[primary["n_reviews_genome"] > 0, "imdb_id"])
    if TEXT_FEATURES.exists():
        feats = pd.read_parquet(TEXT_FEATURES)
        # Recompute if coverage looks stale.
        if set(feats["imdb_id"]) != ids and len(feats) < 0.9 * len(ids):
            feats = compute_movie_text_features(ids, max_per_movie=40, sample_size=100)
    else:
        feats = compute_movie_text_features(ids, max_per_movie=40, sample_size=100)

    genre = genre_sentiment_table(primary, feats)
    genre.to_csv(OUTPUTS / "genre_text_features.csv", index=False)

    genre_null = sentiment_profit_by_genre_null(primary, feats, n_perm=500, random_state=0)
    genre_null.to_csv(OUTPUTS / "genre_sentiment_profit_spearman.csv", index=False)

    # Overall sentiment vs profit / ratings with nulls
    m = primary.merge(feats, on="imdb_id", how="inner")
    checks = {
        "vader_vs_log_profit": spearman_with_null(m["mean_vader_compound"], m["log_gross_over_budget"]),
        "vader_vs_imdb": spearman_with_null(m["mean_vader_compound"], m["imdb_rating"]),
        "diversity_vs_log_profit": spearman_with_null(m["mean_lexical_diversity"], m["log_gross_over_budget"]),
        "length_vs_log_profit": spearman_with_null(m["mean_review_tokens"], m["log_gross_over_budget"]),
    }
    pd.DataFrame(checks).T.to_csv(OUTPUTS / "text_feature_spearman.csv")

    apply_bw_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    top_genres = genre.head(10)
    axes[0].barh(top_genres["genre"][::-1], top_genres["mean_vader"][::-1], color="black")
    axes[0].set_xlabel("Mean VADER compound")
    axes[0].set_ylabel("Genre")
    axes[0].set_title("Sentiment by genre")
    axes[1].barh(top_genres["genre"][::-1], top_genres["mean_diversity"][::-1], color="black")
    axes[1].set_xlabel(f"Mean TTR (n={100} tokens)")
    axes[1].set_title("Lexical diversity")
    scatter_bw(
        axes[2],
        m["mean_vader_compound"],
        m["log_gross_over_budget"],
        "Mean VADER compound",
        "log(gross/budget)",
        title="Sentiment vs money",
        alpha=0.3,
    )
    axes[2].axhline(0.0, color="black", lw=0.8, ls="--")
    save_fig(
        fig,
        OUTPUTS / "fig_text_features_genre.png",
        "Figure. Audience Tag Genome text features for 100k+ movies (year ≤ 2017): "
        "mean VADER compound and fixed-sample lexical diversity by genre (top 10 by "
        "count), and movie-level sentiment vs log(gross/budget).",
    )

    export_hand_label_sample(ids, n=50, random_state=42)
    hand = agreement_from_hand_labels()
    stanford = validate_vader_on_stanford(max_docs=2000)
    meta = {"hand_label": hand, "stanford_vader": stanford, "n_movies_text": int(len(feats))}
    (OUTPUTS / "sentiment_validation.json").write_text(json.dumps(meta, indent=2), encoding="utf8")
    return feats, meta


def idea2(primary: pd.DataFrame) -> pd.DataFrame:
    # Use movies with both critic reception text and genome reviews.
    sub = primary[(primary["critic_text_chars"] > 0) & (primary["n_reviews_genome"] > 0)].copy()
    # Attach title for name blacklist.
    table, critic, audience = run_comparison(sub, min_count=30, max_per_movie=25)
    apply_bw_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        table["critic_per_10k"],
        table["audience_per_10k"],
        c="black",
        alpha=0.35,
        s=16,
        edgecolors="none",
    )
    lims = [
        min(table["critic_per_10k"].min(), table["audience_per_10k"].min()),
        max(table["critic_per_10k"].max(), table["audience_per_10k"].max()),
    ]
    ax.plot(lims, lims, color="black", lw=0.9, ls="--", label="shared rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Critic text: rate per 10,000 words")
    ax.set_ylabel("Audience reviews: rate per 10,000 words")
    ax.set_title("Critic vs audience vocabulary")
    ax.grid(True, which="both")
    # Label extremes off the diagonal.
    table = table.copy()
    table["abs_log_ratio"] = table["log_ratio_audience_over_critic"].abs()
    extremes = table.nlargest(12, "abs_log_ratio")
    for _, row in extremes.iterrows():
        ax.annotate(
            row["term"],
            (row["critic_per_10k"], row["audience_per_10k"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.legend(loc="upper left")
    save_fig(
        fig,
        OUTPUTS / "fig_scattertext_critic_audience.png",
        "Figure. Scattertext-style term rates (per 10,000 words, log axes) in "
        "Wikipedia reception sections vs Tag Genome audience reviews. Dashed "
        "diagonal: equal rate. Person/title tokens removed; min count = 30 in at "
        "least one corpus. Near-diagonal terms are shared language.",
    )
    extremes.to_csv(OUTPUTS / "scattertext_top_divergent_terms.csv", index=False)

    # Inspect texts behind top terms (private snippets only).
    examples = {}
    for term in extremes["term"].head(8):
        examples[term] = top_term_examples(term, critic, audience, n=2, window=70)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "scattertext_term_windows.json").write_text(json.dumps(examples, indent=2), encoding="utf8")
    # Commit-safe: term list only, no windows.
    pd.DataFrame({"term": extremes["term"].head(8).tolist()}).to_csv(
        OUTPUTS / "scattertext_inspected_terms.csv", index=False
    )
    return table


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table = load_or_build_table()
    primary = primary_sample(table, max_year=MAX_YEAR)
    print("primary", len(primary), "under_budget", int(primary["under_budget"].sum()))
    idea1(primary)
    print("idea1 done")
    feats, meta = text_block(primary)
    print("text features", len(feats), meta)
    idea2(primary)
    print("idea2 done; outputs in", OUTPUTS)


if __name__ == "__main__":
    main()
