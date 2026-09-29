from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PRODUCTS = [
    (101, "electronics", 700.0, 0.18),
    (102, "electronics", 1200.0, 0.22),
    (103, "electronics", 450.0, 0.20),
    (201, "home", 850.0, 0.15),
    (202, "home", 1500.0, 0.18),
    (203, "home", 600.0, 0.16),
    (301, "beauty", 350.0, 0.12),
    (302, "beauty", 500.0, 0.14),
    (303, "beauty", 280.0, 0.13),
    (401, "sports", 900.0, 0.17),
    (402, "sports", 1300.0, 0.19),
    (403, "sports", 550.0, 0.15),
]


def build_dataset(start: str = "2023-01-01", end: str = "2025-12-31", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    rows: list[dict] = []

    for product_id, category, base_price, elasticity in PRODUCTS:
        product_factor = 0.85 + rng.random() * 0.35
        base_demand = {"electronics": 70, "home": 55, "beauty": 80, "sports": 60}[category]
        product_demand = base_demand * product_factor
        for date in dates:
            dow = date.dayofweek
            month = date.month
            weekend_factor = 1.10 if dow >= 5 else 0.96
            seasonal_factor = 1.0 + 0.14 * np.sin(2 * np.pi * (month - 1) / 12)
            holiday = int((month == 11 and date.day >= 20) or (month == 12 and date.day <= 31))
            holiday_factor = 1.28 if holiday else 1.0
            promotion = int(rng.random() < (0.14 + 0.08 * holiday))
            discount = float(rng.choice([0, 5, 10, 15, 20], p=[0.52, 0.18, 0.16, 0.09, 0.05])) if promotion else 0.0
            competitor_price = base_price * (1 + rng.normal(0.02, 0.06))
            competitor_price = max(50.0, competitor_price)
            price = base_price * (1 - discount / 100) * (1 + rng.normal(0, 0.025))
            price = max(50.0, price)
            price_effect = (price / base_price) ** (-1.0 - elasticity)
            competitor_effect = 1 + 0.35 * ((competitor_price - price) / base_price)
            promotion_factor = 1.13 if promotion else 1.0
            inventory = int(max(30, rng.normal(420 if category != "beauty" else 500, 80)))
            expected = product_demand * weekend_factor * seasonal_factor * holiday_factor
            expected *= price_effect * competitor_effect * promotion_factor
            noise = rng.normal(0, max(3.0, expected * 0.08))
            units_sold = max(0, int(round(expected + noise)))
            rows.append(
                {
                    "date": date,
                    "product_id": product_id,
                    "category": category,
                    "units_sold": units_sold,
                    "price": round(price, 2),
                    "competitor_price": round(competitor_price, 2),
                    "discount": discount,
                    "promotion": promotion,
                    "inventory": inventory,
                    "holiday": holiday,
                }
            )

    df = pd.DataFrame(rows).sort_values(["product_id", "date"]).reset_index(drop=True)
    return df


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "data" / "raw" / "sales.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    df.to_csv(output, index=False)
    print(f"Wrote {len(df):,} rows to {output}")


if __name__ == "__main__":
    main()
