# Sales Forecast Service

Weekly revenue forecasting for SAP-style sales data. It trains a gradient boosting model, checks it against simple baselines with a time-series backtest, saves a versioned model with its metrics, and serves forecasts through a FastAPI endpoint.

[![CI](https://github.com/Lakshminarayana7844/sales-forecast-service/actions/workflows/ci.yml/badge.svg)](https://github.com/Lakshminarayana7844/sales-forecast-service/actions/workflows/ci.yml)

## Result on the sample data

Expanding-window backtest, 3 folds of 8 weeks, MAPE (lower is better):

| Method | MAPE |
|---|---|
| Gradient boosting (this project) | 5.39 |
| Naive (last week) | 6.66 |
| Seasonal naive (same week last year) | 9.31 |

The sample data (`data/weekly_sales.csv`) is synthetic: 3 regions, 156 weeks, trend, yearly season, a Q4 peak and noise. The point is the workflow, not the number.

## How it works

- `forecast/features.py`: lags (1, 2, 4, 52 weeks), 4-week rolling mean, week-of-year sin/cos, Q4 flag, region code. Every feature uses only past values. A test checks this against the raw history.
- `forecast/model.py`: backtest, training, versioned artifacts (`model_YYYYMMDD.joblib` plus a `.json` with metrics and features, and `latest.json`), and a recursive multi-step forecast.
- `forecast/api.py`: FastAPI service. Trains on first start if no model exists.
- Input validation: duplicate (week, region) rows and negative or missing revenue are rejected.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /model` | version, training date, backtest metrics, features |
| `GET /forecast?region=EMEA&weeks=4` | forecast for 1 to 26 weeks; regions: EMEA, APAC, AMER |

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
python -m forecast --out models          # train and print metrics
uvicorn forecast.api:app --reload
```

Docker (trains at build time):

```bash
docker build -t sales-forecast-service .
docker run -p 8000:8000 sales-forecast-service
```

Use your own data: a CSV with columns `week` (Monday dates), `region`, `revenue`. Set `DATA_PATH` and `MODEL_DIR` to change locations. Regions are listed in `forecast/features.py`.

## Limits

- Recursive forecasting feeds predictions back in, so error grows with the horizon. Keep it to a few weeks for planning use.
- One global model for all regions. Per-region models or hierarchical reconciliation would be the next step.
