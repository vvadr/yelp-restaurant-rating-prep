from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_businesses(path: Path) -> pd.DataFrame:
    """Load Yelp business JSON Lines."""
    return pd.read_json(path, lines=True)


def filter_restaurants(
    businesses: pd.DataFrame,
    city: str,
    open_only: bool = False,
) -> pd.DataFrame:
    """Keep restaurants for one city."""
    categories = businesses["categories"].fillna("")
    mask = businesses["city"].eq(city) & categories.str.contains("Restaurants", case=False)
    if open_only:
        mask &= businesses["is_open"].eq(1)
    return businesses.loc[mask].reset_index(drop=True).copy()


def expand_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Expand the nested attributes dict without index-alignment duplication."""
    clean = df.copy().reset_index(drop=True)
    attributes = clean["attributes"].apply(lambda value: value if isinstance(value, dict) else {})
    attrs = pd.json_normalize(attributes).reset_index(drop=True)
    return pd.concat([clean, attrs], axis=1)


def clean_bool(value: Any) -> int:
    """Convert common Yelp boolean-like values to 0/1."""
    if pd.isna(value):
        return 0
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower().replace("u'", "").replace("'", "")
    if text in {"true", "yes", "1"}:
        return 1
    if text in {"false", "no", "0", "none", "nan"}:
        return 0
    return int("true" in text)


def clean_wifi(value: Any) -> int:
    """Treat free or paid WiFi as available."""
    if pd.isna(value):
        return 0
    text = str(value).strip().lower().replace("u'", "").replace("'", "")
    return int(text in {"free", "paid", "yes", "true"})


def clean_parking(value: Any) -> int:
    """Return 1 when any parking option is true."""
    if pd.isna(value):
        return 0
    if isinstance(value, dict):
        return int(any(clean_bool(v) for v in value.values()))
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return 0
    if isinstance(parsed, dict):
        return int(any(clean_bool(v) for v in parsed.values()))
    return 0


def clean_alcohol(value: Any) -> int:
    text = _clean_text(value)
    mapping = {"none": 0, "beer_and_wine": 1, "full_bar": 2}
    return mapping.get(text, 0)


def clean_noise(value: Any) -> int:
    text = _clean_text(value)
    mapping = {"quiet": 1, "average": 2, "loud": 3, "very_loud": 4}
    return mapping.get(text, 0)


def clean_price(value: Any) -> int:
    if pd.isna(value):
        return 0
    try:
        price = int(float(str(value).strip()))
    except ValueError:
        return 0
    return price if 1 <= price <= 4 else 0


def normalize_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the attribute-derived fields required by the final schema."""
    out = df.copy()
    out["price_range_num"] = out.get("RestaurantsPriceRange2", pd.Series(index=out.index)).apply(clean_price)
    out["has_wifi"] = out.get("WiFi", pd.Series(index=out.index)).apply(clean_wifi)
    out["has_parking"] = out.get("BusinessParking", pd.Series(index=out.index)).apply(clean_parking)
    out["accepts_credit_cards"] = out.get(
        "BusinessAcceptsCreditCards", pd.Series(index=out.index)
    ).apply(clean_bool)
    out["outdoor_seating"] = out.get("OutdoorSeating", pd.Series(index=out.index)).apply(clean_bool)
    out["alcohol_num"] = out.get("Alcohol", pd.Series(index=out.index)).apply(clean_alcohol)
    out["noise_level_num"] = out.get("NoiseLevel", pd.Series(index=out.index)).apply(clean_noise)
    out["n_categories"] = (
        out["categories"]
        .fillna("")
        .apply(lambda value: len([part for part in str(value).split(", ") if part]))
        .astype(int)
    )
    out["review_count"] = out["review_count"].fillna(0).astype(int)
    out["log_review_count"] = np.log1p(out["review_count"]).astype(float)
    return out


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("u'", "").replace("'", "")

