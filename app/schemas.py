from __future__ import annotations

from datetime import date as Date

from pydantic import BaseModel, Field, field_validator


class DemandRequest(BaseModel):
    product_id: int = Field(gt=0)
    category: str = Field(min_length=1)
    price: float = Field(gt=0)
    competitor_price: float = Field(gt=0)
    discount: float = Field(ge=0, le=100)
    promotion: int = Field(ge=0, le=1)
    inventory: int = Field(ge=0)
    holiday: int = Field(ge=0, le=1)
    recent_sales: list[float] = Field(min_length=30)
    date: Date | None = None

    @field_validator("recent_sales")
    @classmethod
    def validate_sales(cls, values: list[float]) -> list[float]:
        if any(value < 0 for value in values):
            raise ValueError("recent_sales cannot contain negative values")
        return values


class DemandResponse(BaseModel):
    product_id: int
    predicted_demand: float


class PriceRecommendationRequest(BaseModel):
    product_id: int = Field(gt=0)
    category: str = Field(min_length=1)
    current_price: float = Field(gt=0)
    competitor_price: float = Field(gt=0)
    discount: float = Field(ge=0, le=100)
    promotion: int = Field(ge=0, le=1)
    inventory: int = Field(ge=0)
    holiday: int = Field(ge=0, le=1)
    unit_cost: float | None = Field(default=None, ge=0)
    minimum_price: float | None = Field(default=None, gt=0)
    maximum_price: float | None = Field(default=None, gt=0)
    recent_sales: list[float] = Field(min_length=30)
    date: Date | None = None

    @field_validator("recent_sales")
    @classmethod
    def validate_sales(cls, values: list[float]) -> list[float]:
        if any(value < 0 for value in values):
            raise ValueError("recent_sales cannot contain negative values")
        return values

    @field_validator("maximum_price")
    @classmethod
    def validate_price_range(cls, value: float | None, info):
        minimum = info.data.get("minimum_price")
        if value is not None and minimum is not None and value < minimum:
            raise ValueError("maximum_price must be greater than or equal to minimum_price")
        return value


class CandidateEvaluation(BaseModel):
    price: float
    predicted_demand: float
    revenue: float
    profit: float | None
    score: float


class PriceRecommendationResponse(BaseModel):
    product_id: int
    current_price: float
    recommended_price: float
    predicted_demand: float
    expected_revenue: float
    expected_profit: float | None
    objective: str
    candidate_prices: list[CandidateEvaluation]
