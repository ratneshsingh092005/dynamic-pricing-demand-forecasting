# E-Commerce Dynamic Pricing & Demand Forecasting Engine

An end-to-end machine learning system that predicts product demand and recommends optimal prices using historical sales, pricing, competitor pricing, promotions, inventory, and seasonal patterns.

The project combines **time-series feature engineering, LightGBM regression, price elasticity analysis, and dynamic pricing optimization**, exposed through a FastAPI backend with a Streamlit frontend.

---

## Features

- Demand prediction using LightGBM
- Time-series lag and rolling features
- Chronological train/test split
- Price elasticity estimation
- Dynamic price recommendation
- Revenue and profit optimization
- FastAPI REST API
- Streamlit frontend
- Pydantic input validation
- Joblib model serialization
- Automated tests

---

## Architecture

```text
Historical Sales Data
        ↓
Feature Engineering
        ↓
Chronological Split
        ↓
LightGBM Demand Model
        ↓
Predicted Demand
        ↓
Candidate Price Generation
        ↓
Revenue / Profit Evaluation
        ↓
Recommended Price
        ↓
FastAPI
        ↓
Streamlit Frontend
```

---

## Dataset

The project uses a synthetic e-commerce dataset containing daily product-level sales data.

| Column | Description |
|---|---|
| `date` | Sales date |
| `product_id` | Product identifier |
| `category` | Product category |
| `units_sold` | Daily demand |
| `price` | Product selling price |
| `competitor_price` | Competitor price |
| `discount` | Discount percentage |
| `promotion` | Promotion indicator |
| `inventory` | Available inventory |
| `holiday` | Holiday indicator |

The dataset contains multiple products across categories including electronics, home, beauty, and sports.

---

## Feature Engineering

### Temporal Features

```text
day_of_week
month
is_weekend
is_holiday
```

### Lag Features

```text
lag_1
lag_7
lag_14
```

### Rolling Features

```text
rolling_mean_7
rolling_mean_30
rolling_std_7
```

### Pricing & Business Features

```text
price
competitor_price
price_difference
discount
promotion
inventory
```

Lag and rolling features are calculated using previous observations to prevent future information from leaking into the model.

---

## Demand Forecasting

**LightGBM Regressor** is used as the primary demand prediction model.

Target:

```text
units_sold
```

A chronological split is used instead of a random train/test split:

```text
Earlier Data                         Later Data
|---------------- Training ----------------|--- Test ---|
                    80%                       20%
```

### Model Performance

| Metric | Score |
|---|---:|
| MAE | 5.13 |
| RMSE | 6.59 |
| MAPE | 7.07% |
| R² | 0.888 |

---

## Price Elasticity

A simple log-price/log-demand regression is used to estimate the relationship between price and demand.

Estimated price elasticity:

```text
-0.276
```

The negative value indicates an inverse association between price and demand in the analyzed data.

This is an observational estimate rather than a causal estimate, since factors such as promotions, seasonality, and competitor pricing can also influence demand.

---

## Dynamic Pricing

The pricing engine evaluates candidate prices around the current price.

For example:

```text
Current Price: ₹1000

Candidates:
₹900
₹950
₹1000
₹1050
₹1100
```

For each candidate price:

1. Required features are generated.
2. LightGBM predicts demand.
3. Expected revenue is calculated.
4. Expected profit is calculated when unit cost is available.
5. The candidate with the highest objective value is selected.

### Revenue

```text
Revenue = Price × Predicted Demand
```

### Profit

```text
Profit = (Price - Unit Cost) × Predicted Demand
```

Minimum and maximum price constraints can also be provided.

---

## API

The backend is built using **FastAPI**.

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

### Predict Demand

```http
POST /predict-demand
```

Example request:

```json
{
  "product_id": 101,
  "category": "electronics",
  "price": 650,
  "competitor_price": 670,
  "discount": 5,
  "promotion": 1,
  "inventory": 400,
  "holiday": 0,
  "recent_sales": [65,68,70,67,72,71,69,73,75,70,68,71,74,76,72,70,69,73,75,77,74,72,71,76,78,75,73,77,79,80]
}
```

Response:

```json
{
  "product_id": 101,
  "predicted_demand": 82.41
}
```

### Recommend Price

```http
POST /recommend-price
```

This endpoint evaluates multiple candidate prices and returns the recommended price along with predicted demand, expected revenue, and expected profit.

---

## Streamlit Frontend

The project includes a lightweight Streamlit interface with two sections.

### Demand Prediction

- Product information
- Pricing information
- Promotion and inventory
- Recent sales
- Predicted demand

### Dynamic Pricing

- Current price
- Unit cost
- Price constraints
- Competitor price
- Recent sales
- Recommended price
- Candidate price evaluation

---

## Project Structure

```text
ecommerce-dynamic-pricing/
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── dependencies.py
│   └── services/
│       ├── demand_service.py
│       └── pricing_service.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── models/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── features/
│   │   └── build_features.py
│   └── pricing/
│       ├── elasticity.py
│       └── optimizer.py
│
├── tests/
│
├── train.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ratneshsingh092005/dynamic-pricing-demand-forecasting.git
cd dynamic-pricing-demand-forecasting
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

This generates the trained LightGBM model:

```text
models/demand_model.joblib
```

---

## Run FastAPI

```bash
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run Streamlit

Install frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

Run:

```bash
streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

---

## Testing

Run the test suite:

```bash
python -m pytest -q
```

Current result:

```text
9 passed
```

Tests cover feature engineering, pricing logic, candidate price generation, API endpoints, and request validation.

---

## Tech Stack

**Python | LightGBM | Pandas | NumPy | Scikit-learn | FastAPI | Streamlit | Pydantic | Joblib | Matplotlib | Seaborn**

---

## Limitations

- Dataset is synthetic and does not represent real customer behavior.
- Price elasticity is observational rather than causal.
- Pricing optimization evaluates a limited candidate-price range.
- External factors such as customer segments and marketing campaigns are not modeled.
- The model currently requires retraining when new historical data becomes available.

---

## Future Improvements

- Real-world sales data integration
- Automated model retraining
- Inventory-aware pricing strategies
- More robust causal elasticity estimation
- Model monitoring and drift detection
- Cloud deployment
- Authentication for production APIs

---


