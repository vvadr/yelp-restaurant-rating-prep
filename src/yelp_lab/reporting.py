from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import AFTER_REVIEWS_FEATURES, AT_LAUNCH_FEATURES, FIGURES_DIR
from .features import top_categories


def create_figures(
    final_df: pd.DataFrame,
    enriched_businesses: pd.DataFrame,
    feature_importance: pd.Series | None,
    figures_dir: Path = FIGURES_DIR,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_stars(final_df, figures_dir / "stars_distribution.png")
    _plot_top_categories(enriched_businesses, figures_dir / "top_categories.png")
    if feature_importance is not None and not feature_importance.empty:
        _plot_feature_importance(feature_importance, figures_dir / "top_feature_importance.png")


def write_findings(
    final_df: pd.DataFrame,
    city: str,
    report_path: Path,
    feature_importance: pd.Series | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    at_launch_count = len(AT_LAUNCH_FEATURES)
    after_reviews_count = len(AFTER_REVIEWS_FEATURES)
    top_features = _format_top_features(feature_importance)

    insights = [
        f"The final {city} restaurant dataset contains {len(final_df):,} rows with a mean star rating of {final_df['stars'].mean():.2f}.",
        f"Restaurants in the middle 50% are open {final_df['hours_open_per_week'].quantile(0.25):.0f}-{final_df['hours_open_per_week'].quantile(0.75):.0f} hours per week.",
        f"The average review text length is {final_df['mean_review_length'].mean():.0f} characters, which gives Module 4/7 useful warm-start signals.",
    ]

    content = f"""# Yelp Module 3 Findings

## Final Numbers

- City: {city}
- Final row count: {len(final_df):,}
- Mean stars: {final_df['stars'].mean():.2f}
- Number of AT-LAUNCH features: {at_launch_count}
- Number of AFTER-REVIEWS features: {after_reviews_count}

## Top 3 Insights

1. {insights[0]}
2. {insights[1]}
3. {insights[2]}

## Feature Signals

{top_features}

## One Question For Module 4

Can a stronger model, such as Random Forest or Gradient Boosting, beat the simple baseline and reduce mean absolute error below about 0.6 stars?

## Summary Chart

See `reports/figures/top_feature_importance.png` for the top feature-importance chart with AT-LAUNCH and AFTER-REVIEWS coloring.
"""
    report_path.write_text(content, encoding="utf-8")


def _plot_stars(df: pd.DataFrame, output_path: Path) -> None:
    counts = df["stars"].value_counts().sort_index()
    ax = counts.plot(kind="bar", color="#2f6f73", figsize=(8, 5))
    ax.set_title("Philadelphia Restaurant Star Ratings")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Number of restaurants")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def _plot_top_categories(df: pd.DataFrame, output_path: Path) -> None:
    counts = top_categories(df, n=15).sort_values()
    ax = counts.plot(kind="barh", color="#bf6b3f", figsize=(9, 6))
    ax.set_title("Top 15 Restaurant Categories")
    ax.set_xlabel("Number of restaurants")
    ax.set_ylabel("Category")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def _plot_feature_importance(feature_importance: pd.Series, output_path: Path) -> None:
    top = feature_importance.sort_values().tail(15)
    colors = [
        "#2f8f4e" if name in AT_LAUNCH_FEATURES else "#d9822b"
        for name in top.index
    ]
    ax = top.plot(kind="barh", color=colors, figsize=(9, 6))
    ax.set_title("Top 15 predictors of star rating - green = available at launch")
    ax.set_xlabel("Random Forest importance")
    ax.set_ylabel("Feature")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def _format_top_features(feature_importance: pd.Series | None) -> str:
    if feature_importance is None or feature_importance.empty:
        return "Feature importance was not generated because scikit-learn was unavailable."
    lines = ["Top feature importance signals:"]
    for name, value in feature_importance.sort_values(ascending=False).head(10).items():
        label = "AT-LAUNCH" if name in AT_LAUNCH_FEATURES else "AFTER-REVIEWS"
        lines.append(f"- {name}: {value:.4f} ({label})")
    return "\n".join(lines)
