# E-Commerce Dynamic Pricing & Demand Forecasting Engine

An end-to-end, interview-explainable ML system that forecasts product demand and evaluates candidate prices using a trained LightGBM regression model.

## Problem statement

E-commerce businesses need to estimate demand before changing a product's price. This project combines historical sales, temporal patterns, promotions, inventory and competitor pricing to:

1. Predict product demand.
2. Quantify the observed relationship between price and demand with a simple elasticity estimate.
3. Evaluate a small set of candidate prices.
4. Choose the candidate that maximizes expected profit when unit cost is available, otherwise expected revenue.
5. Serve the trained model through FastAPI.
6. Run the API in Docker.

The attached project specification explicitly prioritizes correctness and explainability over adding infrastructure such as MLflow, Kubernetes, Kafka, Redis or a database. fileciteturn0file0L13-L29

## Architecture

```mermaid
flowchart LR
    A[Historical sales CSV] --> B[Feature engineering]
    B --> C[Chronological 80/20 split]
    C --> D[LightGBM demand model]
    D --> E[Saved joblib artifact]
    E --> F[FastAPI]
    F --> G[/predict-demand]
    F --> H[/recommend-price]
    H --> I[5 candidate prices]
    I --> D
    I --> J[Revenue / profit calculation]
    J --> K[Recommended price]
```

## Dataset

The repository uses a deliberately sized synthetic e-commerce history because the required schema contains business fields that are not reliably present together in a single small public dataset. It contains 12 products across electronics, home, beauty and sports, with daily observations from 2023-01-01 through 2025-12-31 (13,152 rows).

The generator creates realistic relationships rather than random independent columns: seasonality, weekend effects, promotions, holidays, competitor prices, discounts and a negative price-demand relationship are incorporated into the demand process. The generated dataset is therefore a transparent demonstration dataset, not a claim that it represents a real retailer.

Required fields are `date`, `product_id`, `category`, `units_sold`, `price`, `competitor_price`, `discount`, `promotion`, `inventory` and `holiday`, matching the project specification. fileciteturn0file0L69-L90

## Feature engineering

Features are generated after sorting by product and date:

- Temporal: `day_of_week`, `month`, `is_weekend`, `is_holiday`
- Lags: `lag_1`, `lag_7`, `lag_14`
- Rolling: `rolling_mean_7`, `rolling_mean_30`, `rolling_std_7`
- Pricing: `price`, `competitor_price`, `price_difference`, `discount`
- Business: `promotion`, `inventory`

The rolling features explicitly use `shift(1)` before the rolling window. That means a row cannot use its own demand or future demand to construct a feature. Warm-up rows that do not have the required historical observations are removed.

This directly addresses the specification's requirements around lag features, rolling features and leakage prevention. fileciteturn0file0L112-L149

## Train/test strategy

A chronological split is used: the earliest 80% of dates are training data and the latest 20% are the future/unseen evaluation period. No random `train_test_split` is used because random splitting would allow observations from later periods to appear in training while evaluating earlier observations, which does not represent the intended forecasting scenario.

The final API never retrains a model per request. It loads `models/demand_model.joblib` at startup, as required by the specification. fileciteturn0file0L152-L185

## Model

Primary model: **LightGBM regression** with a small preprocessing pipeline:

- One-hot encoding for `product_id` and `category`.
- Median imputation for numeric features.
- LightGBM `LGBMRegressor`.

Evaluation on the latest 20% period from the generated dataset:

| Metric | Value |
|---|---:|
| MAE | 5.13 |
| RMSE | 6.59 |
| MAPE | 7.07% |
| R² | 0.888 |

Top model features in this run include current price, recent rolling demand, price difference, longer-term rolling demand, inventory and month.

Prophet is intentionally not included in the final architecture. The specification makes it optional and says the final API should use LightGBM; skipping an extra dependency keeps the project easier to explain and deploy. fileciteturn0file0L190-L202

## Price elasticity

A simple log-log linear regression estimates the observed relationship between price and demand:

`log(1 + units_sold) = intercept + elasticity * log(price)`

The current generated-data training run produced an elasticity estimate of approximately **-0.276**. This is an association estimate, not a causal estimate. Promotions, seasonality, competitor prices and product differences can confound the relationship, so it should not be interpreted as a controlled experiment result.

This intentionally avoids Double Machine Learning and other causal-inference complexity, as required by the specification. fileciteturn0file0L205-L220

## Pricing algorithm

The pricing engine uses a deliberately simple candidate search:

For a current price of ₹999, it evaluates approximately:

- ₹899.10
- ₹949.05
- ₹999.00
- ₹1048.95
- ₹1098.90

For every candidate it:

1. Builds the same model features used by training.
2. Predicts demand with LightGBM.
3. Calculates expected revenue: `price × predicted_demand`.
4. If `unit_cost` is supplied, calculates expected profit: `(price - unit_cost) × predicted_demand`.
5. Selects the highest valid objective.

Minimum and maximum price constraints can be supplied. No complex optimization library is required, matching the specification. fileciteturn0file0L223-L277

## API

### `GET /health`

Returns service/model health.

### `POST /predict-demand`

Example:

```json
{
  "product_id": 101,
  "category": "electronics",
  "price": 999,
  "competitor_price": 949,
  "discount": 10,
  "promotion": 1,
  "inventory": 300,
  "holiday": 0,
  "recent_sales": [180,181,182,183,184,185,186,187,180,181,182,183,184,185,186,187,180,181,182,183,184,185,186,187,180,181,182,183,184,185],
  "date": "2025-12-20"
}
```

Example response:

```json
{
  "product_id": 101,
  "predicted_demand": 185.42
}
```

### `POST /recommend-price`

Example:

```json
{
  "product_id": 101,
  "category": "electronics",
  "current_price": 999,
  "competitor_price": 949,
  "discount": 10,
  "promotion": 1,
  "inventory": 300,
  "holiday": 0,
  "unit_cost": 600,
  "minimum_price": 850,
  "maximum_price": 1150,
  "recent_sales": [180,181,182,183,184,185,186,187,180,181,182,183,184,185,186,187,180,181,182,183,184,185,186,187,180,181,182,183,184,185],
  "date": "2025-12-20"
}
```

The response contains the recommended price, predicted demand, expected revenue, expected profit and the individual candidate evaluations.

FastAPI and Pydantic validation reject invalid values such as negative prices or fewer than 30 recent daily sales observations. The endpoint design follows the requested `/health`, `/predict-demand` and `/recommend-price` interface. fileciteturn0file0L280-L342

## Project structure

```text
ecommerce-dynamic-pricing/
├── data/
│   ├── raw/
│   │   └── sales.csv
│   └── processed/
│       └── features.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data/
│   │   └── generate_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   └── train.py
│   └── pricing/
│       ├── elasticity.py
│       └── optimizer.py
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── dependencies.py
│   └── services/
│       ├── demand_service.py
│       └── pricing_service.py
├── models/
│   ├── demand_model.joblib
│   ├── feature_importance.csv
│   └── metadata.json
├── tests/
│   ├── test_features.py
│   ├── test_pricing.py
│   └── test_api.py
├── train.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Run locally

### 1. Install dependencies

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This generates the dataset, features, evaluation metrics and model artifacts.

### 3. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open Swagger at `http://localhost:8000/docs`.

### 4. Run tests

```bash
pytest -q
```

## Docker

Build:

```bash
docker build -t ecommerce-pricing .
```

Run:

```bash
docker run -p 8000:8000 ecommerce-pricing
```

Or:

```bash
docker compose up --build
```

The image trains the model during the image build, so the container starts with the required model artifact and does not retrain for each API request. The requested Docker workflow is `docker build -t ecommerce-pricing .` followed by `docker run -p 8000:8000 ecommerce-pricing`. fileciteturn0file0L418-L429

## Limitations

- The dataset is synthetic and therefore cannot establish real-world business performance.
- The elasticity estimate is observational, not causal.
- Candidate prices are limited to five points around the current price.
- The model assumes recent demand history is available at serving time.
- Inventory is treated as an input feature, not as a full stockout simulation.
- There is no online learning or automatic retraining.
- Forecast accuracy can degrade when market conditions change.
- Revenue/profit estimates are model estimates, not guarantees.

## Future improvements

If the project later needs more production complexity, reasonable next steps would be model monitoring, automated retraining, richer causal price experimentation, a real transactional data source, uncertainty estimates and more sophisticated inventory constraints. These are deliberately outside the current scope.

## Interview explanation in one minute

> “I built a demand forecasting and dynamic pricing engine for e-commerce. I first sort sales chronologically and create lag and rolling features without using future demand. I use a time-based 80/20 split instead of a random split, then train LightGBM to predict units sold. For pricing, I evaluate five candidate prices around the current price, run each through the demand model, calculate expected revenue or profit, and choose the best valid candidate. I also calculate a simple log-log price elasticity as an interpretable analysis, while clearly treating it as an observational relationship rather than a causal estimate. The trained pipeline is serialized with Joblib and served through FastAPI in Docker.”
