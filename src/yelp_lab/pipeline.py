from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cleaning import (
    expand_attributes,
    filter_restaurants,
    load_businesses,
    normalize_business_features,
)
from .config import (
    BUSINESS_JSON,
    DEFAULT_CHUNKSIZE,
    DEFAULT_CITY,
    FINAL_DATASET,
    FINAL_SCHEMA,
    FINDINGS_REPORT,
    INTERIM_DIR,
    PIPELINE_MODEL,
    REVIEW_JSON,
)
from .features import add_engineered_features
from .reporting import create_figures, write_findings
from .reviews import aggregate_reviews, merge_review_features
from .validation import coerce_final_types, validate_final_schema


@dataclass(frozen=True)
class PipelineResult:
    final_dataset: Path
    findings_report: Path
    row_count: int
    feature_importance_created: bool


def build_final_dataset(
    city: str = DEFAULT_CITY,
    open_only: bool = False,
    business_path: Path = BUSINESS_JSON,
    review_path: Path = REVIEW_JSON,
    output_path: Path = FINAL_DATASET,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    businesses = load_businesses(business_path)
    restaurants = filter_restaurants(businesses, city=city, open_only=open_only)
    restaurants.to_parquet(INTERIM_DIR / "business_step1.parquet", index=False)

    enriched = expand_attributes(restaurants)
    enriched = normalize_business_features(enriched)
    enriched = add_engineered_features(enriched, city=city)
    enriched.to_parquet(INTERIM_DIR / "business_step2.parquet", index=False)

    review_features = aggregate_reviews(
        review_path=review_path,
        business_ids=set(enriched["business_id"]),
        chunksize=chunksize,
    )
    enriched = merge_review_features(enriched, review_features)
    enriched.to_parquet(INTERIM_DIR / "business_step3.parquet", index=False)

    final_df = coerce_final_types(enriched[FINAL_SCHEMA])
    validate_final_schema(final_df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    return final_df, enriched


def run_pipeline(
    city: str = DEFAULT_CITY,
    open_only: bool = False,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> PipelineResult:
    final_df, enriched = build_final_dataset(city=city, open_only=open_only, chunksize=chunksize)
    feature_importance = train_optional_model(final_df)
    create_figures(final_df, enriched, feature_importance)
    write_findings(final_df, city=city, report_path=FINDINGS_REPORT, feature_importance=feature_importance)
    return PipelineResult(
        final_dataset=FINAL_DATASET,
        findings_report=FINDINGS_REPORT,
        row_count=len(final_df),
        feature_importance_created=feature_importance is not None,
    )


def train_optional_model(final_df: pd.DataFrame) -> pd.Series | None:
    """Train a small reproducible baseline and return feature importances."""
    try:
        import joblib
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError:
        return None

    X = final_df.drop(columns=["stars", "business_id"])
    y = final_df["stars"]
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )
    model = RandomForestRegressor(n_estimators=75, max_depth=8, n_jobs=-1, random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics_path = INTERIM_DIR / "model_metrics.txt"
    metrics_path.write_text(
        f"MAE: {mean_absolute_error(y_test, y_pred):.4f}\n"
        f"R2: {r2_score(y_test, y_pred):.4f}\n",
        encoding="utf-8",
    )

    PIPELINE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_MODEL)

    transformed_names = preprocessor.get_feature_names_out()
    readable_names = [_readable_feature_name(name) for name in transformed_names]
    importance = pd.Series(model.feature_importances_, index=readable_names)
    return importance.groupby(level=0).sum().sort_values(ascending=False)


def _readable_feature_name(transformed_name: str) -> str:
    """Map ColumnTransformer output names back to the original feature names."""
    if transformed_name.startswith("num__"):
        return transformed_name.removeprefix("num__")
    if transformed_name.startswith("cat__"):
        raw_name = transformed_name.removeprefix("cat__")
        for column in ("state", "city"):
            prefix = f"{column}_"
            if raw_name.startswith(prefix):
                return column
        return raw_name
    return transformed_name
