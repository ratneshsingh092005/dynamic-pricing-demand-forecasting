from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.dependencies import load_model
from app.schemas import DemandRequest, DemandResponse, PriceRecommendationRequest, PriceRecommendationResponse
from app.services.demand_service import DemandService
from app.services.pricing_service import PricingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    app.state.demand_service = DemandService(model)
    app.state.pricing_service = PricingService(model)
    yield


app = FastAPI(title="E-Commerce Dynamic Pricing & Demand Forecasting Engine", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "demand_service")}


@app.post("/predict-demand", response_model=DemandResponse)
def predict_demand(payload: DemandRequest, request: Request):
    try:
        demand = request.app.state.demand_service.predict(payload)
        return {"product_id": payload.product_id, "predicted_demand": round(demand, 2)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/recommend-price", response_model=PriceRecommendationResponse)
def recommend_price(payload: PriceRecommendationRequest, request: Request):
    try:
        return request.app.state.pricing_service.recommend(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
