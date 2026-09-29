import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def payload():
    return {
        "product_id": 101,
        "category": "electronics",
        "price": 999,
        "competitor_price": 949,
        "discount": 10,
        "promotion": 1,
        "inventory": 300,
        "holiday": 0,
        "recent_sales": [180 + i % 8 for i in range(30)],
        "date": "2025-12-20",
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_demand_prediction(client):
    response = client.post("/predict-demand", json=payload())
    assert response.status_code == 200
    assert response.json()["predicted_demand"] >= 0


def test_invalid_recent_sales(client):
    data = payload()
    data["recent_sales"] = [1, 2]
    response = client.post("/predict-demand", json=data)
    assert response.status_code == 422


def test_price_recommendation(client):
    data = payload()
    data.update({"current_price": 999, "unit_cost": 600})
    data.pop("price")
    response = client.post("/recommend-price", json=data)
    assert response.status_code == 200
    body = response.json()
    assert body["recommended_price"] > 0
    assert len(body["candidate_prices"]) == 5
