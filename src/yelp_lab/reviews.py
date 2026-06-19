from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "amazing",
    "awesome",
    "delicious",
    "tasty",
    "fresh",
    "friendly",
    "nice",
    "perfect",
    "best",
    "love",
    "loved",
    "wonderful",
    "fantastic",
    "happy",
    "clean",
    "fast",
}

NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "horrible",
    "worst",
    "slow",
    "rude",
    "dirty",
    "cold",
    "bland",
    "expensive",
    "overpriced",
    "disappointed",
    "poor",
    "hate",
    "hated",
    "gross",
    "nasty",
}


def simple_sentiment_score(text: object) -> int:
    if not isinstance(text, str):
        return 0
    words = re.findall(r"\b[a-z]+\b", text.lower())
    positive = sum(1 for word in words if word in POSITIVE_WORDS)
    negative = sum(1 for word in words if word in NEGATIVE_WORDS)
    return positive - negative


def aggregate_reviews(
    review_path: Path,
    business_ids: set[str],
    chunksize: int,
) -> pd.DataFrame:
    """Read review JSON in chunks and aggregate features by business_id."""
    aggregates: list[pd.DataFrame] = []
    columns = ["business_id", "text", "date"]

    for chunk in pd.read_json(review_path, lines=True, chunksize=chunksize):
        keep = chunk.loc[chunk["business_id"].isin(business_ids), columns].copy()
        if keep.empty:
            continue

        keep["review_length"] = keep["text"].str.len().fillna(0)
        keep["sentiment_score"] = keep["text"].apply(simple_sentiment_score)
        keep["date"] = pd.to_datetime(keep["date"], errors="coerce")

        grouped = keep.groupby("business_id").agg(
            review_rows=("business_id", "size"),
            mean_review_length=("review_length", "mean"),
            mean_review_sentiment_score=("sentiment_score", "mean"),
            first_review_date=("date", "min"),
            last_review_date=("date", "max"),
        )
        aggregates.append(grouped)

    if not aggregates:
        return _empty_review_features()

    combined = pd.concat(aggregates).reset_index()
    final = combined.groupby("business_id").agg(
        review_rows=("review_rows", "sum"),
        mean_review_length=("mean_review_length", _weighted_average_factory(combined, "review_rows")),
        mean_review_sentiment_score=(
            "mean_review_sentiment_score",
            _weighted_average_factory(combined, "review_rows"),
        ),
        first_review_date=("first_review_date", "min"),
        last_review_date=("last_review_date", "max"),
    )
    final = final.reset_index()
    span_days = (final["last_review_date"] - final["first_review_date"]).dt.days.clip(lower=365)
    final["review_velocity"] = final["review_rows"] / (span_days / 365.25)
    return final[
        [
            "business_id",
            "review_velocity",
            "mean_review_length",
            "mean_review_sentiment_score",
        ]
    ]


def merge_review_features(businesses: pd.DataFrame, review_features: pd.DataFrame) -> pd.DataFrame:
    out = businesses.merge(review_features, on="business_id", how="left")
    for column in ["review_velocity", "mean_review_length", "mean_review_sentiment_score"]:
        out[column] = out[column].fillna(0).astype(float)
    return out


def _empty_review_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "business_id",
            "review_velocity",
            "mean_review_length",
            "mean_review_sentiment_score",
        ]
    )


def _weighted_average_factory(source: pd.DataFrame, weight_column: str):
    weights_by_index = source[weight_column]

    def weighted_average(values: pd.Series) -> float:
        weights = weights_by_index.loc[values.index]
        total_weight = weights.sum()
        if total_weight == 0:
            return 0.0
        return float(np.average(values, weights=weights))

    return weighted_average

