from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


SETUP_CODE = r"""
from pathlib import Path
import os
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Image, display

from yelp_lab.config import FINAL_SCHEMA

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "yelp_restaurants_clean.parquet"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "classes"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_DISPLAY_DIR = PROJECT_ROOT / ".cache" / "notebook_figures"
NOTEBOOK_DISPLAY_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
_display_counter = 0

def show_and_close():
    global _display_counter
    for number in plt.get_fignums():
        _display_counter += 1
        fig = plt.figure(number)
        display_path = NOTEBOOK_DISPLAY_DIR / f"figure_{_display_counter:03d}.png"
        fig.savefig(display_path, dpi=160, bbox_inches="tight")
        display(Image(filename=str(display_path)))
    plt.close("all")

plt.show = show_and_close

print(df.shape)
df.head()
"""


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def notebook(title: str, cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        md(
            f"""
# {title}

Yelp Restaurants Module 3 lab notebook. This notebook is organized for GitHub review and uses the final clean dataset produced by `scripts/run_pipeline.py`.
"""
        ),
        code(SETUP_CODE),
        *cells,
    ]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    return nb


def save_and_execute(filename: str, nb: nbf.NotebookNode) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / filename
    client = NotebookClient(nb, timeout=240, kernel_name="python3", resources={"metadata": {"path": str(PROJECT_ROOT)}})
    client.execute()
    nbf.write(nb, path)
    print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


def build_class_1() -> nbf.NotebookNode:
    return notebook(
        "Class 1 - Data Cleaning",
        [
            md(
                """
## Goal

Filter Yelp business data to restaurants in one city, understand the target, and save the first clean business table.

Original PDF tasks:
- Explore city, state, and star-rating distributions.
- Filter to restaurants in one city.
- Save `business_step1.parquet`.
- Make 3+ exploratory charts and 1 explanatory chart for Priya.

Note: raw Yelp JSON files are not committed to GitHub. The reproducible raw-to-final implementation lives in `src/yelp_lab/` and `scripts/run_pipeline.py`; this notebook uses the final clean parquet for reviewer-friendly evidence.
"""
            ),
            code(
                r"""
print("Rows:", len(df))
print("City:", df["city"].unique().tolist())
print("State:", df["state"].unique().tolist())
print("Columns:", len(df.columns))
df[["business_id", "city", "state", "stars", "review_count"]].head()
"""
            ),
            md("## Exploratory Chart 1 - Star Rating Distribution"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(8, 5))
df["stars"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#2f6f73")
ax.set_title("Exploratory: Distribution of restaurant star ratings")
ax.set_xlabel("Stars")
ax.set_ylabel("Restaurant count")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class1_exploratory_stars.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 2 - Review Count Shape"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["log_review_count"], bins=30, color="#6f5f90", edgecolor="white")
ax.set_title("Exploratory: Log review count distribution")
ax.set_xlabel("log(1 + review_count)")
ax.set_ylabel("Restaurant count")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class1_exploratory_reviews.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 3 - Restaurant Locations"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(df["longitude"], df["latitude"], c=df["stars"], cmap="RdYlGn", alpha=0.65, s=18)
ax.set_title("Exploratory: Philadelphia restaurants by location and stars")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
fig.colorbar(scatter, ax=ax, label="Stars")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class1_exploratory_location.png", dpi=160)
plt.show()
"""
            ),
            md("## Explanatory Chart for Priya - Most Restaurants Cluster Around 3.5-4.0 Stars"),
            code(
                r"""
rating_summary = df.groupby("stars").size().rename("restaurant_count").reset_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(rating_summary["stars"].astype(str), rating_summary["restaurant_count"], color="#bf6b3f")
ax.axvline(str(df["stars"].median()), color="black", linestyle="--", label=f"Median = {df['stars'].median():.1f}")
ax.set_title("Priya: Philadelphia restaurant ratings are concentrated near 3.5-4.0 stars")
ax.set_xlabel("Star rating")
ax.set_ylabel("Number of restaurants")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class1_explanatory_rating_concentration.png", dpi=160)
plt.show()
print("Takeaway: Most restaurants are not extreme; the target is centered near the middle-high range.")
"""
            ),
            md(
                """
## Self-Check

- Final city is Philadelphia.
- Final row count is in the expected 5k-10k range.
- Star distribution has been inspected.
- Location and review-count patterns have been inspected.
"""
            ),
        ],
    )


def build_class_2() -> nbf.NotebookNode:
    return notebook(
        "Class 2 - Encoding and Scaling",
        [
            md(
                """
## Goal

Convert messy categorical/business attributes into model-ready numbers and inspect their distributions.

Original PDF tasks:
- Expand nested attributes.
- Build numeric encodings for price, WiFi, parking, credit cards, outdoor seating, alcohol, and noise.
- Scale numeric columns.
- Make 3+ exploratory charts and 1 explanatory chart.
"""
            ),
            code(
                r"""
encoded_cols = [
    "price_range_num", "has_wifi", "has_parking", "accepts_credit_cards",
    "outdoor_seating", "alcohol_num", "noise_level_num"
]
df[encoded_cols].describe().T
"""
            ),
            md("## Exploratory Chart 1 - Price Range Counts"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(8, 5))
df["price_range_num"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#375a7f")
ax.set_title("Exploratory: Encoded price range counts")
ax.set_xlabel("Price range number (0 = missing)")
ax.set_ylabel("Restaurant count")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class2_exploratory_price.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 2 - Boolean Attribute Coverage"),
            code(
                r"""
bool_cols = ["has_wifi", "has_parking", "accepts_credit_cards", "outdoor_seating"]
coverage = df[bool_cols].mean().sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
coverage.plot(kind="barh", ax=ax, color="#2f8f4e")
ax.set_title("Exploratory: Share of restaurants with each encoded attribute")
ax.set_xlabel("Share with value = 1")
ax.set_ylabel("Feature")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class2_exploratory_attribute_coverage.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 3 - Scaled Numeric Features"),
            code(
                r"""
from sklearn.preprocessing import StandardScaler

scale_cols = ["price_range_num", "n_categories", "hours_open_per_week", "dist_from_center", "log_review_count"]
scaled = pd.DataFrame(StandardScaler().fit_transform(df[scale_cols]), columns=scale_cols)
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=scaled, ax=ax, color="#d8c99b")
ax.set_title("Exploratory: Scaled numeric feature spread")
ax.set_xlabel("Feature")
ax.set_ylabel("Standardized value")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class2_exploratory_scaled_features.png", dpi=160)
plt.show()
"""
            ),
            md("## Explanatory Chart for Priya - Price Range and Stars"),
            code(
                r"""
price_stars = df.groupby("price_range_num")["stars"].mean()
fig, ax = plt.subplots(figsize=(8, 5))
price_stars.plot(kind="bar", ax=ax, color="#bf6b3f")
ax.set_title("Priya: Average stars by encoded price range")
ax.set_xlabel("Price range number (0 = missing)")
ax.set_ylabel("Mean stars")
ax.set_ylim(3.0, 4.1)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class2_explanatory_price_vs_stars.png", dpi=160)
plt.show()
print("Takeaway: Encoded price is usable as a model feature, but it is not enough by itself to explain ratings.")
"""
            ),
            md(
                """
## Self-Check

- Required attribute encodings exist.
- Missing/unknown values are represented numerically.
- Numeric scaling was demonstrated.
- Price/attribute signals were charted for Priya.
"""
            ),
        ],
    )


def build_class_3() -> nbf.NotebookNode:
    return notebook(
        "Class 3 - Feature Engineering",
        [
            md(
                """
## Goal

Create features that describe hours, geography, categories, and reviews.

Original PDF tasks:
- Compute `hours_open_per_week` and `n_days_open`.
- Compute Haversine distance from city center.
- Compute category counts.
- Read reviews in chunks and build review-based features.
- Make 3+ exploratory charts and 1 explanatory chart.
"""
            ),
            code(
                r"""
feature_cols = [
    "hours_open_per_week", "n_days_open", "dist_from_center",
    "n_categories", "review_velocity", "mean_review_length",
    "mean_review_sentiment_score"
]
df[feature_cols].describe().T
"""
            ),
            md("## Exploratory Chart 1 - Hours Open Per Week"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["hours_open_per_week"], bins=30, color="#2f6f73", edgecolor="white")
ax.set_title("Exploratory: Hours open per week")
ax.set_xlabel("Hours open per week")
ax.set_ylabel("Restaurant count")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class3_exploratory_hours.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 2 - Distance From City Center"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["dist_from_center"], df["stars"], alpha=0.35, color="#6f5f90", s=18)
ax.set_title("Exploratory: Distance from center vs stars")
ax.set_xlabel("Distance from Philadelphia center (km)")
ax.set_ylabel("Stars")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class3_exploratory_distance.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 3 - Review Sentiment Signal"),
            code(
                r"""
sample = df.sample(min(1500, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sample["mean_review_sentiment_score"], sample["stars"], alpha=0.35, color="#bf6b3f", s=18)
ax.set_title("Exploratory: Mean review sentiment vs stars")
ax.set_xlabel("Mean review sentiment score")
ax.set_ylabel("Stars")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class3_exploratory_sentiment.png", dpi=160)
plt.show()
"""
            ),
            md("## Explanatory Chart for Priya - Hours Open and Star Ratings"),
            code(
                r"""
hours_bins = pd.cut(
    df["hours_open_per_week"],
    bins=[-0.1, 20, 40, 60, 90, 120, 168],
    labels=["0-20", "21-40", "41-60", "61-90", "91-120", "121-168"],
)
hours_summary = df.groupby(hours_bins, observed=True)["stars"].mean()
fig, ax = plt.subplots(figsize=(8, 5))
hours_summary.plot(kind="bar", ax=ax, color="#375a7f")
ax.set_title("Priya: Average stars by weekly opening-hours band")
ax.set_xlabel("Hours open per week")
ax.set_ylabel("Mean stars")
ax.set_ylim(3.0, 4.1)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class3_explanatory_hours_vs_stars.png", dpi=160)
plt.show()
print("Takeaway: Hours matter, but they should be combined with location, metadata, and review signals.")
"""
            ),
            md(
                """
## Self-Check

- Hours features exist and are numeric.
- Distance from center exists.
- Category count exists.
- Review velocity, review length, and sentiment features exist.
"""
            ),
        ],
    )


def build_class_4() -> nbf.NotebookNode:
    return notebook(
        "Class 4 - Feature Selection",
        [
            md(
                """
## Goal

Reduce the engineered dataset to a useful set of predictive columns and explain which features matter most.

Original PDF tasks:
- Split train/test before scoring features.
- Inspect correlations.
- Compute mutual information.
- Train Random Forest feature importance as a second opinion.
- Mark AT-LAUNCH vs AFTER-REVIEWS features.
- Make 3+ exploratory charts and 1 explanatory chart.
"""
            ),
            code(
                r"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

feature_df = df.drop(columns=["business_id", "stars"])
feature_df = pd.get_dummies(feature_df, columns=["state", "city"], drop_first=False)
X = feature_df
y = df["stars"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
"""
            ),
            md("## Exploratory Chart 1 - Correlation Heatmap"),
            code(
                r"""
corr_cols = [
    "latitude", "longitude", "dist_from_center", "n_categories", "price_range_num",
    "hours_open_per_week", "n_days_open", "review_count", "log_review_count",
    "review_velocity", "mean_review_length", "mean_review_sentiment_score", "stars"
]
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(df[corr_cols].corr(), cmap="vlag", center=0, ax=ax)
ax.set_title("Exploratory: Numeric feature correlation heatmap")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class4_exploratory_correlation_heatmap.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 2 - Mutual Information Scores"),
            code(
                r"""
mi = pd.Series(mutual_info_regression(X_train, y_train, random_state=42), index=X_train.columns)
mi_top = mi.sort_values().tail(15)
fig, ax = plt.subplots(figsize=(9, 6))
mi_top.plot(kind="barh", ax=ax, color="#2f6f73")
ax.set_title("Exploratory: Top mutual information scores")
ax.set_xlabel("Mutual information")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class4_exploratory_mutual_info.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 3 - Random Forest Importances"),
            code(
                r"""
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
rf_importance.tail(15).plot(kind="barh", ax=ax, color="#6f5f90")
ax.set_title("Exploratory: Random Forest feature importances")
ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class4_exploratory_rf_importance.png", dpi=160)
plt.show()
"""
            ),
            md("## Explanatory Chart for Priya - Top 15 Predictors"),
            code(
                r"""
at_launch = {
    "state_PA", "city_Philadelphia", "latitude", "longitude", "dist_from_center",
    "n_categories", "price_range_num", "has_wifi", "has_parking",
    "accepts_credit_cards", "outdoor_seating", "alcohol_num",
    "noise_level_num", "hours_open_per_week", "n_days_open"
}
top = rf_importance.tail(15)
colors = ["#2f8f4e" if name in at_launch else "#d9822b" for name in top.index]
fig, ax = plt.subplots(figsize=(9, 6))
top.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Priya: Top 15 predictors - green = available at launch")
ax.set_xlabel("Random Forest importance")
ax.set_ylabel("Feature")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class4_explanatory_top_predictors.png", dpi=160)
plt.show()
print("Takeaway: Review-based features are strongest, but launch-time metadata still provides usable cold-start signals.")
"""
            ),
            md(
                """
## Selected Feature Notes

Kept features are the 22 final-schema columns. Features based on reviews are marked AFTER-REVIEWS because they are not available for a brand-new restaurant. Location, hours, price, and attributes are AT-LAUNCH features.
"""
            ),
        ],
    )


def build_class_5() -> nbf.NotebookNode:
    return notebook(
        "Class 5 - Pipelines",
        [
            md(
                """
## Goal

Put preprocessing and modeling into one reproducible scikit-learn Pipeline.

Original PDF tasks:
- Split numeric and categorical columns.
- Build numeric and categorical mini-pipelines.
- Combine them with `ColumnTransformer`.
- Train and evaluate a Ridge regression model.
- Make predicted-vs-actual and residual charts.
"""
            ),
            code(
                r"""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

X = df.drop(columns=["business_id", "stars"])
y = df["stars"]
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = [col for col in X.columns if col not in numeric_cols]
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)
"""
            ),
            code(
                r"""
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0)),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.3f}")
print(f"R2: {r2:.3f}")
"""
            ),
            md("## Exploratory Chart 1 - Predicted vs Actual Stars"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.35, color="#2f6f73", s=18)
ax.plot([1, 5], [1, 5], "r--", label="Perfect prediction")
ax.set_title("Exploratory: Predicted vs actual stars")
ax.set_xlabel("Actual stars")
ax.set_ylabel("Predicted stars")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class5_exploratory_predicted_vs_actual.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 2 - Residual Plot"),
            code(
                r"""
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred, residuals, alpha=0.35, color="#6f5f90", s=18)
ax.axhline(0, color="red", linestyle="--")
ax.set_title("Exploratory: Residuals by predicted stars")
ax.set_xlabel("Predicted stars")
ax.set_ylabel("Residual (actual - predicted)")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class5_exploratory_residuals.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 3 - Absolute Error Distribution"),
            code(
                r"""
abs_errors = np.abs(residuals)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(abs_errors, bins=30, color="#bf6b3f", edgecolor="white")
ax.set_title("Exploratory: Absolute prediction error distribution")
ax.set_xlabel("Absolute error in stars")
ax.set_ylabel("Restaurant count")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class5_exploratory_absolute_error.png", dpi=160)
plt.show()
"""
            ),
            md("## Explanatory Chart for Priya - How Close Are Predictions?"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.35, color="#375a7f", s=18)
ax.plot([1, 5], [1, 5], "r--")
ax.set_title(f"Priya: Predicted vs actual stars - MAE = {mae:.2f}")
ax.set_xlabel("Actual stars")
ax.set_ylabel("Predicted stars")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class5_explanatory_prediction_quality.png", dpi=160)
plt.show()
print(f"Takeaway: The baseline is off by about {mae:.2f} stars on average. This is better than unknown, and Module 4 can improve it.")
"""
            ),
        ],
    )


def build_class_6() -> nbf.NotebookNode:
    return notebook(
        "Class 6 - End-to-End Lab",
        [
            md(
                """
## Goal

Validate the final 22-column dataset and package the findings for Priya.

Original PDF tasks:
- Produce `yelp_restaurants_clean.parquet`.
- Validate the exact 22-column schema.
- Reuse the most important charts.
- Write `findings.md`.
"""
            ),
            code(
                r"""
assert list(df.columns) == FINAL_SCHEMA
assert df["business_id"].duplicated().sum() == 0
assert df["stars"].between(1, 5).all()
print("Final schema validated.")
print("Rows:", len(df))
print("Mean stars:", round(df["stars"].mean(), 3))
df.dtypes
"""
            ),
            md("## Required Final Schema"),
            code(
                r"""
pd.DataFrame({"column_number": range(1, len(FINAL_SCHEMA) + 1), "column_name": FINAL_SCHEMA})
"""
            ),
            md("## Exploratory Chart 1 - Target Distribution"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(8, 5))
df["stars"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#2f6f73")
ax.set_title("Class 6: Final target distribution")
ax.set_xlabel("Stars")
ax.set_ylabel("Restaurant count")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class6_exploratory_target_distribution.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 2 - Final Feature Correlations"),
            code(
                r"""
fig, ax = plt.subplots(figsize=(11, 8))
sns.heatmap(df.select_dtypes(include=["number"]).corr(), cmap="vlag", center=0, ax=ax)
ax.set_title("Class 6: Correlation heatmap for final numeric features")
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class6_exploratory_final_correlations.png", dpi=160)
plt.show()
"""
            ),
            md("## Exploratory Chart 3 - Launch vs Review Feature Groups"),
            code(
                r"""
launch_features = [
    "dist_from_center", "n_categories", "price_range_num", "has_wifi",
    "has_parking", "accepts_credit_cards", "outdoor_seating", "alcohol_num",
    "noise_level_num", "hours_open_per_week", "n_days_open"
]
review_features = ["review_count", "log_review_count", "review_velocity", "mean_review_length", "mean_review_sentiment_score"]
summary = pd.Series({
    "AT-LAUNCH features": len(launch_features),
    "AFTER-REVIEWS features": len(review_features),
})
fig, ax = plt.subplots(figsize=(7, 5))
summary.plot(kind="bar", ax=ax, color=["#2f8f4e", "#d9822b"])
ax.set_title("Class 6: Feature availability groups")
ax.set_ylabel("Feature count")
ax.tick_params(axis="x", rotation=0)
fig.tight_layout()
fig.savefig(FIGURE_DIR / "class6_exploratory_feature_groups.png", dpi=160)
plt.show()
"""
            ),
            md("## Explanatory Chart for Priya - Final Summary"),
            code(
                r"""
importance_path = PROJECT_ROOT / "reports" / "figures" / "top_feature_importance.png"
if importance_path.exists():
    from IPython.display import Image, display
    display(Image(filename=str(importance_path)))
print("Takeaway: Review-based features are strongest, but the project clearly separates launch-time and after-review signals for future cold-start modeling.")
"""
            ),
            md(
                """
## Final Submission Checklist

- `data/processed/yelp_restaurants_clean.parquet` exists.
- `reports/findings.md` exists.
- Class notebooks include exploratory and explanatory charts.
- Raw Yelp files are intentionally excluded from GitHub.
"""
            ),
        ],
    )


def main() -> None:
    notebooks = {
        "01_class1_data_cleaning.ipynb": build_class_1(),
        "02_class2_encoding_scaling.ipynb": build_class_2(),
        "03_class3_feature_engineering.ipynb": build_class_3(),
        "04_class4_feature_selection.ipynb": build_class_4(),
        "05_class5_pipelines.ipynb": build_class_5(),
        "06_class6_end_to_end_lab.ipynb": build_class_6(),
    }
    for filename, nb in notebooks.items():
        save_and_execute(filename, nb)


if __name__ == "__main__":
    main()
