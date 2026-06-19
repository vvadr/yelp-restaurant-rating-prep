from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CITY_CENTERS


def hours_for_one_day(time_string: Any) -> float:
    """Return open hours for a Yelp time span like '11:0-22:0'."""
    if not isinstance(time_string, str) or "-" not in time_string:
        return 0.0
    open_str, close_str = time_string.split("-", maxsplit=1)
    try:
        open_h, open_m = [int(part) for part in open_str.split(":")]
        close_h, close_m = [int(part) for part in close_str.split(":")]
    except ValueError:
        return 0.0

    open_time = open_h + open_m / 60
    close_time = close_h + close_m / 60
    if open_time == 0 and close_time == 0:
        return 24.0
    if close_time < open_time:
        close_time += 24
    return max(close_time - open_time, 0.0)


def hours_open_per_week(hours_dict: Any) -> float:
    if not isinstance(hours_dict, dict):
        return 0.0
    return float(sum(hours_for_one_day(value) for value in hours_dict.values()))


def n_days_open(hours_dict: Any) -> int:
    if not isinstance(hours_dict, dict):
        return 0
    return int(sum(1 for value in hours_dict.values() if value))


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: float, lon2: float) -> pd.Series:
    radius_km = 6371.0
    phi1 = np.radians(lat1.astype(float))
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1.astype(float))
    dlambda = np.radians(lon2 - lon1.astype(float))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def add_engineered_features(df: pd.DataFrame, city: str) -> pd.DataFrame:
    out = df.copy()
    center_lat, center_lon = CITY_CENTERS.get(
        city,
        (float(out["latitude"].median()), float(out["longitude"].median())),
    )
    out["hours_open_per_week"] = out["hours"].apply(hours_open_per_week).astype(float)
    out["n_days_open"] = out["hours"].apply(n_days_open).astype(int)
    out["dist_from_center"] = haversine_km(
        out["latitude"],
        out["longitude"],
        center_lat,
        center_lon,
    ).astype(float)
    return out


def top_categories(df: pd.DataFrame, n: int = 15) -> pd.Series:
    return df["categories"].fillna("").str.split(", ").explode().value_counts().head(n)

