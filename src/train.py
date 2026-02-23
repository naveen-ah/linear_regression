from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df.rename(columns={"MedHouseVal": "target"}, inplace=True)
    return df


def run_eda(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "description": df.describe().to_dict(),
        "correlation_with_target": df.corr(numeric_only=True)["target"].sort_values(ascending=False).to_dict(),
    }

    with open(REPORTS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(11, 7))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.histplot(df["target"], kde=True, bins=40)
    plt.title("Target Distribution")
    plt.xlabel("Median House Value")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "target_distribution.png", dpi=180)
    plt.close()


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["rooms_per_person"] = X["AveRooms"] / np.clip(X["AveOccup"], a_min=1e-5, a_max=None)
    X["bedrooms_per_room"] = X["AveBedrms"] / np.clip(X["AveRooms"], a_min=1e-5, a_max=None)
    X["population_per_household"] = X["Population"] / np.clip(X["AveOccup"], a_min=1e-5, a_max=None)
    X["distance_to_coast_proxy"] = np.sqrt(X["Latitude"] ** 2 + X["Longitude"] ** 2)
    return X


def build_pipeline(feature_names: list[str]) -> GridSearchCV:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_names)],
        remainder="drop",
    )

    base_pipeline = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocessor", preprocessor),
            ("feature_selection", SelectFromModel(estimator=LassoCV(cv=5, random_state=RANDOM_STATE), threshold="median")),
            ("model", Ridge(random_state=RANDOM_STATE)),
        ]
    )

    param_grid = [
        {
            "model": [LinearRegression()],
        },
        {
            "model": [Ridge(random_state=RANDOM_STATE)],
            "model__alpha": [0.1, 1.0, 5.0, 10.0],
        },
        {
            "model": [ElasticNet(random_state=RANDOM_STATE, max_iter=5000)],
            "model__alpha": [0.01, 0.1, 1.0],
            "model__l1_ratio": [0.2, 0.5, 0.8],
        },
    ]

    model_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        verbose=1,
    )

    return model_search


def evaluate(model: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)

    metrics = {
        "rmse": float(mean_squared_error(y_test, predictions, squared=False)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "best_params": model.best_params_,
        "best_cv_score_rmse": float(-model.best_score_),
    }
    return metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    run_eda(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    feature_names = list(add_engineered_features(X_train.head(3)).columns)
    model_search = build_pipeline(feature_names)
    model_search.fit(X_train, y_train)

    metrics = evaluate(model_search, X_test, y_test)

    joblib.dump(model_search.best_estimator_, MODELS_DIR / "linear_regression_pipeline.joblib")
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
