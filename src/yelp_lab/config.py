from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BUSINESS_JSON = RAW_DIR / "yelp_academic_dataset_business.json"
REVIEW_JSON = RAW_DIR / "yelp_academic_dataset_review.json"

FINAL_DATASET = PROCESSED_DIR / "yelp_restaurants_clean.parquet"
PIPELINE_MODEL = PROCESSED_DIR / "yelp_pipeline.joblib"
FINDINGS_REPORT = REPORTS_DIR / "findings.md"

DEFAULT_CITY = "Philadelphia"
DEFAULT_CHUNKSIZE = 100_000

CITY_CENTERS = {
    "Philadelphia": (39.9526, -75.1652),
    "Tampa": (27.9506, -82.4572),
    "Indianapolis": (39.7684, -86.1581),
    "Nashville": (36.1627, -86.7816),
    "Tucson": (32.2226, -110.9747),
}

FINAL_SCHEMA = [
    "business_id",
    "state",
    "city",
    "latitude",
    "longitude",
    "dist_from_center",
    "n_categories",
    "price_range_num",
    "has_wifi",
    "has_parking",
    "accepts_credit_cards",
    "outdoor_seating",
    "alcohol_num",
    "noise_level_num",
    "hours_open_per_week",
    "n_days_open",
    "review_count",
    "log_review_count",
    "review_velocity",
    "mean_review_length",
    "mean_review_sentiment_score",
    "stars",
]

AT_LAUNCH_FEATURES = {
    "state",
    "city",
    "latitude",
    "longitude",
    "dist_from_center",
    "n_categories",
    "price_range_num",
    "has_wifi",
    "has_parking",
    "accepts_credit_cards",
    "outdoor_seating",
    "alcohol_num",
    "noise_level_num",
    "hours_open_per_week",
    "n_days_open",
}

AFTER_REVIEWS_FEATURES = {
    "review_count",
    "log_review_count",
    "review_velocity",
    "mean_review_length",
    "mean_review_sentiment_score",
}

