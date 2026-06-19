from __future__ import annotations

import pandas as pd

from .config import FINAL_SCHEMA


def validate_final_schema(df: pd.DataFrame) -> None:
    missing = [column for column in FINAL_SCHEMA if column not in df.columns]
    extra = [column for column in df.columns if column not in FINAL_SCHEMA]
    if missing:
        raise ValueError(f"Missing final schema columns: {missing}")
    if extra:
        raise ValueError(f"Unexpected final schema columns: {extra}")
    if df["business_id"].duplicated().any():
        duplicated = int(df["business_id"].duplicated().sum())
        raise ValueError(f"Final dataset has {duplicated} duplicated business_id values")
    if not df["stars"].between(1.0, 5.0).all():
        raise ValueError("Final dataset has stars outside the 1.0-5.0 range")


def coerce_final_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    string_columns = ["business_id", "state", "city"]
    int_columns = [
        "n_categories",
        "price_range_num",
        "has_wifi",
        "has_parking",
        "accepts_credit_cards",
        "outdoor_seating",
        "alcohol_num",
        "noise_level_num",
        "n_days_open",
        "review_count",
    ]
    float_columns = [
        "latitude",
        "longitude",
        "dist_from_center",
        "hours_open_per_week",
        "log_review_count",
        "review_velocity",
        "mean_review_length",
        "mean_review_sentiment_score",
        "stars",
    ]

    for column in string_columns:
        out[column] = out[column].astype("string")
    for column in int_columns:
        out[column] = out[column].fillna(0).astype(int)
    for column in float_columns:
        out[column] = out[column].fillna(0).astype(float)
    return out[FINAL_SCHEMA]

