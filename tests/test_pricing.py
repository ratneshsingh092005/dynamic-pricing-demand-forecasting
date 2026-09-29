from src.pricing.elasticity import generate_candidate_prices
from src.pricing.optimizer import recent_sales_features


def test_candidate_prices():
    assert generate_candidate_prices(1000) == [900.0, 950.0, 1000.0, 1050.0, 1100.0]


def test_candidate_price_constraints():
    candidates = generate_candidate_prices(1000, minimum=950, maximum=1050)
    assert candidates == [950.0, 1000.0, 1050.0]


def test_recent_sales_features():
    values = list(range(1, 31))
    features = recent_sales_features(values)
    assert features["lag_1"] == 30
    assert features["lag_7"] == 24
    assert features["lag_14"] == 17
    assert features["rolling_mean_7"] == 27
