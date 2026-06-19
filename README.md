# Yelp Restaurants Module 3 Lab

This project turns the Yelp Open Dataset business and review JSON files into the final Module 3 restaurant dataset:

`data/processed/yelp_restaurants_clean.parquet`

The lab reference is stored in `docs/03_yelp_restaurants_m3_lab.pdf`. Google Drive and download steps from the PDF are skipped because the raw files are expected to exist locally in `data/raw/`.

## Project Layout

```text
data/raw/          Local Yelp JSON/tar files, ignored by Git
data/interim/      Intermediate parquet files, ignored by Git
data/processed/    Final clean parquet output
docs/              Lab reference PDF
reports/           Findings report and generated figures
notebooks/         Executed class-by-class lab notebooks
scripts/           Runnable project commands
src/yelp_lab/      Reusable pipeline code
```

## Setup

Create or activate a Python environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

The raw Yelp files are not included in this repo. To regenerate the final dataset, place the Yelp JSON files in `data/raw/`.

## Run

From the project root:

```powershell
python scripts/run_pipeline.py
```

Default behavior:

- City: `Philadelphia`
- Restaurant filter: categories containing `Restaurants`
- Includes both open and closed restaurants
- Reads reviews in chunks
- Writes the final 22-column parquet dataset
- Writes `reports/findings.md`
- Writes figures to `reports/figures/`

Optional examples:

```powershell
python scripts/run_pipeline.py --city Tampa
python scripts/run_pipeline.py --city Philadelphia --open-only
python scripts/run_pipeline.py --review-chunksize 50000
```

## Class Notebooks

The `notebooks/` folder contains executed notebooks for every PDF class:

- Class 1: Data Cleaning
- Class 2: Encoding and Scaling
- Class 3: Feature Engineering
- Class 4: Feature Selection
- Class 5: Pipelines
- Class 6: End-to-End Lab

Each notebook includes the class goal, tasks, 3+ exploratory charts, and 1 explanatory chart for Priya. To regenerate them:

```powershell
python scripts/create_class_notebooks.py
```

## Final Schema

The final dataset contains exactly these columns:

```text
business_id, state, city, latitude, longitude, dist_from_center,
n_categories, price_range_num, has_wifi, has_parking,
accepts_credit_cards, outdoor_seating, alcohol_num, noise_level_num,
hours_open_per_week, n_days_open, review_count, log_review_count,
review_velocity, mean_review_length, mean_review_sentiment_score, stars
```

## Notes

Large Yelp raw files and interim outputs are ignored by Git. The final cleaned parquet is small enough to keep in the repo with the code, docs, findings report, and figures.
