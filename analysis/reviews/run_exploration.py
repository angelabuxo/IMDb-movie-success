"""Compute review-analysis outputs used by 01_exploration.ipynb.

Writes only derived numbers / figure files under outputs/. Review text stays in
build/imdb/features/reviews_private/ (gitignored via build/).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.reviews.build_movie_table import build_movie_table, primary_sample, save_movie_table
from analysis.reviews.disagreement import (
    add_disagreement_residual,
    disagreement_summary,
    export_top_disagreements,
)
from analysis.reviews.nulls import correlation_table, genre_decade_strata, spearman_with_null
from analysis.reviews.paths import OUTPUTS, PRIVATE, TEXT_FEATURES
from analysis.reviews.plotting import apply_bw_style, annotate_extremes, save_fig, scatter_bw
from analysis.reviews.scattertext_compare import run_disagreement_comparison, top_term_examples
from analysis.reviews.text_features import (
    agreement_from_hand_labels,
    compute_movie_text_features,
    export_hand_label_sample,
    genre_sentiment_table,
    sentiment_profit_by_genre_null,
    validate_vader_on_stanford,
)

MAX_YEAR = 2017
MIN_TERM_COUNT = 25


def load_or_build_table() -> pd.DataFrame:
    table = build_movie_table(max_year=MAX_YEAR)
    return table


def attach_disagreement(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute residual disagreement on 100k+ ≤2017; merge onto full table; save."""
    primary = primary_sample(table, max_year=MAX_YEAR)
    primary = add_disagreement_residual(primary)
    export_top_disagreements(primary, top_n=25)
    summary = disagreement_summary(primary)
    print("disagreement summary")
    print(summary.to_string())

    merge_cols = [
        "imdb_id",
        "rank_log_gross_over_budget",
        "rank_imdb_rating",
        "disagreement_residual",
        "profit_quintile",
        "disagreement_group",
    ]
    # Drop any previous disagreement columns before merge.
    drop = [c for c in merge_cols if c != "imdb_id" and c in table.columns]
    table = table.drop(columns=drop, errors="ignore")
    table = table.merge(primary[merge_cols], on="imdb_id", how="left")
    save_movie_table(table)
    return table, primary


def idea1(primary: pd.DataFrame) -> dict:
    """Spearman with stratified nulls; residual disagreement; extreme profit quintiles."""
    apply_bw_style()
    cols = {
        "log_gross_over_budget": "log(gross/budget)",
        "imdb_rating": "IMDb averageRating",
        "metacritic": "Metacritic",
        "rotten_tomatoes": "Rotten Tomatoes %",
    }
    pair_cols = list(cols)
    strata = genre_decade_strata(primary)
    corr = correlation_table(primary, pair_cols, n_perm=1000, random_state=0, strata=strata)
    corr.to_csv(OUTPUTS / "spearman_correlations.csv", index=False)
    print("spearman (stratified null primary)")
    print(corr[["a", "b", "n", "rho", "null_mean_strat", "null_std_strat", "z_strat", "p_null_strat", "p_null_global"]].to_string(index=False))

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
    caption = (
        "Figure. Pairwise relationships among log(gross/budget), IMDb rating, "
        "Metacritic and Rotten Tomatoes for the 100k+ clean theatrical sample "
        f"with year ≤ {MAX_YEAR}. Associations are tested with a genre×decade "
        "stratified label shuffle (global shuffle reported as secondary)."
    )
    save_fig(fig, OUTPUTS / "fig_scatter_money_scores.png", caption)

    d = primary.dropna(subset=["log_gross_over_budget", "imdb_rating", "disagreement_residual"]).copy()
    above, below = export_top_disagreements(d, top_n=25)
    print("top rating_above_money")
    print(above[["title", "year", "log_gross_over_budget", "imdb_rating", "disagreement_residual"]].head(10).to_string(index=False))
    print("top rating_below_money")
    print(below[["title", "year", "log_gross_over_budget", "imdb_rating", "disagreement_residual"]].head(10).to_string(index=False))

    # Extreme profit quintiles (Q1 / Q5) instead of break-even × median.
    q1 = d[d["profit_quintile"] == 1]
    q5 = d[d["profit_quintile"] == 5]
    fig, ax = plt.subplots(figsize=(7, 6))
    mid = d[d["profit_quintile"].isin([2, 3, 4])]
    ax.scatter(mid["log_gross_over_budget"], mid["imdb_rating"], c="#AAAAAA", alpha=0.25, s=14, edgecolors="none", label="profit Q2–Q4")
    ax.scatter(q1["log_gross_over_budget"], q1["imdb_rating"], c="black", marker="o", alpha=0.55, s=22, label="profit Q1 (lowest)")
    ax.scatter(q5["log_gross_over_budget"], q5["imdb_rating"], c="black", marker="^", alpha=0.55, s=28, label="profit Q5 (highest)")
    # Mark top residual disagreements
    highlight = pd.concat([above.head(8), below.head(8)])
    ax.scatter(
        highlight["log_gross_over_budget"],
        highlight["imdb_rating"],
        facecolors="none",
        edgecolors="black",
        s=70,
        linewidths=1.2,
        marker="s",
        label="top |residual|",
    )
    annotate_extremes(ax, highlight, "log_gross_over_budget", "imdb_rating", "title", n=8)
    ax.set_xlabel("log(gross/budget)")
    ax.set_ylabel("IMDb averageRating")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=8)
    q1_max = float(q1["log_gross_over_budget"].max()) if len(q1) else float("nan")
    q5_min = float(q5["log_gross_over_budget"].min()) if len(q5) else float("nan")
    q_caption = (
        "Figure. Money vs IMDb rating with extreme profit quintiles (100k+ clean, "
        f"year ≤ {MAX_YEAR}). Circles: lowest profit quintile (Q1, log(g/b) ≤ {q1_max:.3f}); "
        f"triangles: highest (Q5, log(g/b) ≥ {q5_min:.3f}). Squares: largest "
        "|disagreement_residual| from rank(IMDb) ~ rank(log(g/b))."
    )
    save_fig(fig, OUTPUTS / "fig_profit_quintiles_disagreement.png", q_caption)

    bias = {
        "n_primary": int(len(primary)),
        "n_under_budget": int(primary["under_budget"].sum()),
        "frac_under_budget": float(primary["under_budget"].mean()),
        "n_with_residual": int(d["disagreement_residual"].notna().sum()),
        "n_profit_q1": int(len(q1)),
        "n_profit_q5": int(len(q5)),
        "profit_q1_max_log": q1_max,
        "profit_q5_min_log": q5_min,
        "mean_imdb_q1": float(q1["imdb_rating"].mean()) if len(q1) else np.nan,
        "mean_imdb_q5": float(q5["imdb_rating"].mean()) if len(q5) else np.nan,
        "n_complete_score_rows": int(primary[list(cols)].dropna(how="any").shape[0]),
    }
    pd.Series(bias).to_csv(OUTPUTS / "idea1_summary.csv", header=["value"])
    return {"corr": corr, "bias": bias}


def text_block(primary: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ids = set(primary.loc[primary["n_reviews_genome"] > 0, "imdb_id"])
    if TEXT_FEATURES.exists():
        feats = pd.read_parquet(TEXT_FEATURES)
        if len(feats) < 0.9 * len(ids):
            feats = compute_movie_text_features(ids, max_per_movie=40, sample_size=100)
    else:
        feats = compute_movie_text_features(ids, max_per_movie=40, sample_size=100)

    genre = genre_sentiment_table(primary, feats)
    genre.to_csv(OUTPUTS / "genre_text_features.csv", index=False)

    genre_null = sentiment_profit_by_genre_null(primary, feats, n_perm=500, random_state=0)
    genre_null.to_csv(OUTPUTS / "genre_sentiment_profit_spearman.csv", index=False)

    m = primary.merge(feats, on="imdb_id", how="inner")
    strata = genre_decade_strata(m)
    checks = {
        "vader_vs_log_profit": spearman_with_null(
            m["mean_vader_compound"], m["log_gross_over_budget"], strata=strata
        ),
        "vader_vs_imdb": spearman_with_null(m["mean_vader_compound"], m["imdb_rating"], strata=strata),
        "diversity_vs_log_profit": spearman_with_null(
            m["mean_lexical_diversity"], m["log_gross_over_budget"], strata=strata
        ),
        "length_vs_log_profit": spearman_with_null(
            m["mean_review_tokens"], m["log_gross_over_budget"], strata=strata
        ),
    }
    pd.DataFrame(checks).T.to_csv(OUTPUTS / "text_feature_spearman.csv")
    print("text feature spearman")
    print(pd.DataFrame(checks).T[["n", "rho", "null_mean_strat", "null_std_strat", "z_strat", "p_null_strat", "p_null_global"]].to_string())

    apply_bw_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    top_genres = genre.head(10)
    axes[0].barh(top_genres["genre"][::-1], top_genres["mean_vader"][::-1], color="black")
    axes[0].set_xlabel("Mean VADER compound")
    axes[0].set_ylabel("Genre")
    axes[0].set_title("Sentiment by genre")
    axes[1].barh(top_genres["genre"][::-1], top_genres["mean_diversity"][::-1], color="black")
    axes[1].set_xlabel("Mean TTR (n=100 tokens)")
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
    """Scattertext: Tag Genome reviews for residual disagreement groups."""
    sub = primary[primary["disagreement_group"].isin(["rating_above_money", "rating_below_money"])].copy()
    table, above_texts, below_texts = run_disagreement_comparison(
        sub, min_count=MIN_TERM_COUNT, max_per_movie=30
    )
    apply_bw_style()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        table["rating_below_money_per_10k"],
        table["rating_above_money_per_10k"],
        c="black",
        alpha=0.35,
        s=16,
        edgecolors="none",
    )
    lims = [
        min(table["rating_below_money_per_10k"].min(), table["rating_above_money_per_10k"].min()),
        max(table["rating_below_money_per_10k"].max(), table["rating_above_money_per_10k"].max()),
    ]
    ax.plot(lims, lims, color="black", lw=0.9, ls="--", label="shared rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rating_below_money: rate per 10,000 words")
    ax.set_ylabel("rating_above_money: rate per 10,000 words")
    ax.set_title("Tag Genome vocabulary by disagreement group")
    ax.grid(True, which="both")
    table = table.copy()
    table["abs_log_ratio"] = table["log_ratio_above_over_below"].abs()
    extremes = table.nlargest(12, "abs_log_ratio")
    for _, row in extremes.iterrows():
        ax.annotate(
            row["term"],
            (row["rating_below_money_per_10k"], row["rating_above_money_per_10k"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.legend(loc="upper left")
    cov = pd.read_csv(OUTPUTS / "scattertext_coverage.csv", index_col=0)["value"]
    n_a = int(float(cov.get("n_movies_with_text_a", 0)))
    n_b = int(float(cov.get("n_movies_with_text_b", 0)))
    save_fig(
        fig,
        OUTPUTS / "fig_scattertext_disagreement_groups.png",
        "Figure. Scattertext-style term rates (per 10,000 words, log axes) in Tag "
        f"Genome reviews for disagreement residual extremes: rating_above_money "
        f"(n={n_a} movies with text) vs rating_below_money (n={n_b}). Dashed "
        f"diagonal: equal rate. Person/title tokens removed; min count = {MIN_TERM_COUNT} "
        "in both corpora.",
    )
    extremes.to_csv(OUTPUTS / "scattertext_top_divergent_terms.csv", index=False)
    print("scattertext coverage")
    print(pd.read_csv(OUTPUTS / "scattertext_coverage.csv", index_col=0).to_string())
    print("top divergent terms")
    print(extremes[["term", "rating_above_money_per_10k", "rating_below_money_per_10k", "log_ratio_above_over_below"]].to_string(index=False))

    examples = {}
    for term in extremes["term"].head(8):
        examples[term] = top_term_examples(term, above_texts, below_texts, n=2, window=70)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    (PRIVATE / "scattertext_term_windows.json").write_text(json.dumps(examples, indent=2), encoding="utf8")
    pd.DataFrame({"term": extremes["term"].head(8).tolist()}).to_csv(
        OUTPUTS / "scattertext_inspected_terms.csv", index=False
    )
    return table


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    table = load_or_build_table()
    table, primary = attach_disagreement(table)
    print("primary", len(primary), "under_budget", int(primary["under_budget"].sum()))
    print(
        "movie_level_table disagreement non-null",
        int(table["disagreement_residual"].notna().sum()),
    )
    idea1(primary)
    print("idea1 done")
    feats, meta = text_block(primary)
    print("text features", len(feats), meta)
    idea2(primary)
    print("idea2 done; outputs in", OUTPUTS)


if __name__ == "__main__":
    main()
