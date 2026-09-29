from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def estimate_price_elasticity(df: pd.DataFrame) -> float:
    """Estimate simple price elasticity from log(price) and log(demand).

    This is an association estimate, not a causal treatment effect. Promotions,
    seasonality, competitor prices and product differences can confound it.
    """
    sample = df[["price", "units_sold"]].copy()
    sample = sample[(sample["price"] > 0) & (sample["units_sold"] >= 0)]
    X = np.log(sample[["price"]])
    y = np.log1p(sample["units_sold"])
    model = LinearRegression().fit(X, y)
    return float(model.coef_[0])


def generate_candidate_prices(current_price: float, minimum: float | None = None, maximum: float | None = None) -> list[float]:
    if current_price <= 0:
        raise ValueError("current_price must be positive")
    candidates = [current_price * factor for factor in (0.90, 0.95, 1.00, 1.05, 1.10)]
    if minimum is not None:
        candidates = [max(minimum, price) for price in candidates]
    if maximum is not None:
        candidates = [min(maximum, price) for price in candidates]
    return sorted({round(price, 2) for price in candidates if price > 0})
