from __future__ import annotations

import pandas as pd

from src.pricing.optimizer import build_inference_row


class DemandService:
    def __init__(self, model) -> None:
        self.model = model

    def predict(self, request) -> float:
        date = pd.Timestamp(request.date or pd.Timestamp.now().date())
        row = build_inference_row(request.model_dump(), request.price, date)
        return max(0.0, float(self.model.predict(row)[0]))
