from __future__ import annotations

import pandas as pd

from src.features.build_features import MODEL_FEATURES
from src.pricing.elasticity import generate_candidate_prices


def recent_sales_features(recent_sales: list[float]) -> dict[str, float]:
    """Build lag and rolling features from recent sales.

    recent_sales must be ordered from oldest to newest.
    The last value represents the most recent completed day's sales.
    """
    if len(recent_sales) < 30:
        raise ValueError("recent_sales must contain at least 30 daily observations")

    values = pd.Series(recent_sales, dtype=float)

    return {
        "lag_1": float(values.iloc[-1]),
        "lag_7": float(values.iloc[-7]),
        "lag_14": float(values.iloc[-14]),
        "rolling_mean_7": float(values.iloc[-7:].mean()),
        "rolling_mean_30": float(values.iloc[-30:].mean()),
        "rolling_std_7": float(values.iloc[-7:].std(ddof=1)),
    }


def build_inference_row(
    request: dict,
    price: float,
    date: pd.Timestamp,
) -> pd.DataFrame:
    recent = recent_sales_features(request["recent_sales"])

    day_of_week = int(date.dayofweek)

    row = {
        "product_id": request["product_id"],
        "category": request["category"],
        "day_of_week": day_of_week,
        "month": int(date.month),
        "is_weekend": int(day_of_week >= 5),
        "is_holiday": int(request["holiday"]),
        **recent,
        "price": float(price),
        "competitor_price": float(request["competitor_price"]),
        "price_difference": float(
            price - request["competitor_price"]
        ),
        "discount": float(request["discount"]),
        "promotion": int(request["promotion"]),
        "inventory": float(request["inventory"]),
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def recommend_price(
    model,
    request: dict,
) -> dict:
    current_price = float(request["current_price"])
    minimum = request.get("minimum_price")
    maximum = request.get("maximum_price")

    candidates = generate_candidate_prices(
        current_price,
        minimum,
        maximum,
    )

    date = pd.Timestamp(
        request.get("date") or pd.Timestamp.now().date()
    )

    unit_cost = request.get("unit_cost")

    evaluations = []

    for candidate in candidates:
        features = build_inference_row(
            request,
            candidate,
            date,
        )

        demand = max(
            0.0,
            float(model.predict(features)[0]),
        )

        revenue = candidate * demand

        profit = (
            (candidate - float(unit_cost)) * demand
            if unit_cost is not None
            else None
        )

        score = profit if profit is not None else revenue

        evaluations.append(
            {
                "price": candidate,
                "predicted_demand": demand,
                "revenue": revenue,
                "profit": profit,
                "score": score,
            }
        )

    best = max(
        evaluations,
        key=lambda item: item["score"],
    )

    return {
        "current_price": current_price,
        "recommended_price": best["price"],
        "predicted_demand": round(
            best["predicted_demand"],
            2,
        ),
        "expected_revenue": round(
            best["revenue"],
            2,
        ),
        "expected_profit": (
            round(best["profit"], 2)
            if best["profit"] is not None
            else None
        ),
        "objective": (
            "profit"
            if unit_cost is not None
            else "revenue"
        ),
        "candidate_prices": evaluations,
    }