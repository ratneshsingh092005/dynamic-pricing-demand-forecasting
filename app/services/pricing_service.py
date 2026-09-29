from __future__ import annotations

from src.pricing.optimizer import recommend_price


class PricingService:
    def __init__(self, model) -> None:
        self.model = model

    def recommend(self, request) -> dict:
        result = recommend_price(
            self.model,
            request.model_dump(),
        )
        result["product_id"] = request.product_id
        return result