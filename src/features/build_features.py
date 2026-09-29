from __future__ import annotations

import pandas as pd


TEMPORAL_FEATURES = ["day_of_week", "month", "is_weekend", "is_holiday"]
LAG_FEATURES = ["lag_1", "lag_7", "lag_14"]
ROLLING_FEATURES = ["rolling_mean_7", "rolling_mean_30", "rolling_std_7"]
PRICE_FEATURES = ["price", "competitor_price", "price_difference", "discount"]
BUSINESS_FEATURES = ["promotion", "inventory"]
MODEL_FEATURES = [
    "product_id",
    "category",
    *TEMPORAL_FEATURES,
    *LAG_FEATURES,
    *ROLLING_FEATURES,
    *PRICE_FEATURES,
    *BUSINESS_FEATURES,
]
TARGET = "units_sold"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date", "product_id", "category", "units_sold", "price",
        "competitor_price", "discount", "promotion", "inventory", "holiday",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["product_id", "date"]).reset_index(drop=True)
    grouped = result.groupby("product_id", sort=False)["units_sold"]

    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    result["is_holiday"] = result["holiday"].astype(int)
    result["lag_1"] = grouped.shift(1)
    result["lag_7"] = grouped.shift(7)
    result["lag_14"] = grouped.shift(14)
    result["rolling_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).mean())
    result["rolling_mean_30"] = grouped.transform(lambda s: s.shift(1).rolling(30).mean())
    result["rolling_std_7"] = grouped.transform(lambda s: s.shift(1).rolling(7).std())
    result["price_difference"] = result["price"] - result["competitor_price"]
    return result


def drop_feature_warmup_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=LAG_FEATURES + ROLLING_FEATURES).reset_index(drop=True)
