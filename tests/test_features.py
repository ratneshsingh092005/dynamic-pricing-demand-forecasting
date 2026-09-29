import pandas as pd

from src.features.build_features import add_features


def test_lags_do_not_use_current_demand():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=16),
        "product_id": [1] * 16,
        "category": ["x"] * 16,
        "units_sold": list(range(1, 17)),
        "price": [100] * 16,
        "competitor_price": [100] * 16,
        "discount": [0] * 16,
        "promotion": [0] * 16,
        "inventory": [100] * 16,
        "holiday": [0] * 16,
    })
    result = add_features(df)
    assert result.loc[7, "lag_1"] == 7
    assert result.loc[7, "lag_7"] == 1
    assert result.loc[7, "rolling_mean_7"] == 4


def test_price_difference():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=2),
        "product_id": [1, 1], "category": ["x", "x"], "units_sold": [10, 11],
        "price": [100, 105], "competitor_price": [95, 100], "discount": [0, 0],
        "promotion": [0, 0], "inventory": [100, 100], "holiday": [0, 0],
    })
    assert add_features(df).loc[1, "price_difference"] == 5
