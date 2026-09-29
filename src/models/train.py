from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features.build_features import MODEL_FEATURES, TARGET, add_features, drop_feature_warmup_rows
from src.pricing.elasticity import estimate_price_elasticity


CATEGORICAL = ["product_id", "category"]
NUMERIC = [c for c in MODEL_FEATURES if c not in CATEGORICAL]


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ],
        remainder="drop",
    )
    model = LGBMRegressor(
        objective="regression",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    preprocessor.set_output(transform="pandas")
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def chronological_split(
    df: pd.DataFrame, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].unique())
    cutoff = dates[int(len(dates) * train_fraction)]

    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    return train, test


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = np.maximum(0, model.predict(X_test))
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    non_zero = y_test != 0
    mape = float(np.mean(np.abs((y_test[non_zero] - predictions[non_zero]) / y_test[non_zero])) * 100)
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE_percent": mape, "R2": float(r2_score(y_test, predictions))}


def feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    return pd.DataFrame({"feature": names, "importance": estimator.feature_importances_}).sort_values("importance", ascending=False)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data" / "raw" / "sales.csv"
    if not raw_path.exists():
        from src.data.generate_dataset import main as generate
        generate()

    df = pd.read_csv(raw_path)
    features = drop_feature_warmup_rows(add_features(df))
    processed_path = root / "data" / "processed" / "features.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(processed_path, index=False)

    train_df, test_df = chronological_split(features)
    X_train, y_train = train_df[MODEL_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[MODEL_FEATURES], test_df[TARGET]

    model = build_pipeline()
    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    importance = feature_importance(model)
    elasticity = estimate_price_elasticity(train_df)

    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / "demand_model.joblib")
    importance.to_csv(model_dir / "feature_importance.csv", index=False)
    metadata = {
        "features": MODEL_FEATURES,
        "target": TARGET,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_end": str(train_df["date"].max().date()),
        "test_start": str(test_df["date"].min().date()),
        "metrics": metrics,
        "price_elasticity": elasticity,
        "objective": "regression",
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete")
    print(json.dumps(metrics, indent=2))
    print(f"Estimated price elasticity: {elasticity:.3f}")
    print("Top features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
