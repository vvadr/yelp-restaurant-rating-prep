from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(SRC_DIR))

from yelp_lab.config import DEFAULT_CHUNKSIZE, DEFAULT_CITY
from yelp_lab.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Yelp Module 3 final dataset.")
    parser.add_argument("--city", default=DEFAULT_CITY, help="City to filter, default: Philadelphia.")
    parser.add_argument(
        "--open-only",
        action="store_true",
        help="Keep only businesses where is_open == 1. Default keeps all restaurants.",
    )
    parser.add_argument(
        "--review-chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="Rows per review JSON chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        city=args.city,
        open_only=args.open_only,
        chunksize=args.review_chunksize,
    )
    print(f"Created {result.final_dataset} with {result.row_count:,} rows.")
    print(f"Wrote {result.findings_report}.")
    if result.feature_importance_created:
        print("Created model metrics and feature-importance chart.")


if __name__ == "__main__":
    main()
