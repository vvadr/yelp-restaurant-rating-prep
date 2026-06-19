# Module 3 Lab Completion Checklist

This project now includes the final pipeline plus class-by-class notebooks for the PDF lab.

## Class 1 - Data Cleaning

- Loads the final restaurant dataset for review.
- Documents the original raw JSON filtering task.
- Includes 3 exploratory charts.
- Includes 1 explanatory chart for Priya.

## Class 2 - Encoding and Scaling

- Reviews encoded business attributes.
- Demonstrates numeric scaling.
- Includes 3 exploratory charts.
- Includes 1 explanatory chart for Priya.

## Class 3 - Feature Engineering

- Reviews hours, distance, category-count, and review-based features.
- Includes 3 exploratory charts.
- Includes 1 explanatory chart for Priya.

## Class 4 - Feature Selection

- Performs train/test split.
- Creates correlation, mutual information, and Random Forest importance charts.
- Marks launch-time vs after-review signal types.
- Includes 1 explanatory top-feature chart for Priya.

## Class 5 - Pipelines

- Builds a scikit-learn preprocessing and Ridge regression pipeline.
- Evaluates MAE and R2.
- Includes predicted-vs-actual, residual, and error charts.
- Includes 1 explanatory prediction-quality chart for Priya.

## Class 6 - End-to-End Lab

- Validates the exact 22-column final schema.
- Confirms row count and target range.
- Reuses the most important final charts.
- Points to `reports/findings.md`.

## Important Note

The raw Yelp JSON files are intentionally not committed because they are too large for GitHub. The notebooks use the final clean parquet committed in `data/processed/`, and the source pipeline can regenerate that parquet when raw files are placed in `data/raw/`.
