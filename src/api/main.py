#RetailPulse REST API – predictions, forecasts, and health checks.
from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.config.loader import get_settings
from src.data.ingestion import DataIngestion
from src.utils.io import load_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)
app = FastAPI(
    title="RetailPulse API",
    description="AI-Powered Customer Analytics & Demand Forecasting",
    version="1.0.0",
)

REQUEST_COUNT = Counter(
    "retailpulse_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "retailpulse_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)
MODEL_METRIC_GAUGE = Gauge(
    "retailpulse_model_metric",
    "RetailPulse model quality metric",
    ["model", "metric"],
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.url.path
    elapsed = time.perf_counter() - start
    REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, endpoint).observe(elapsed)
    return response


def _refresh_model_metric_gauges() -> None:
    settings = get_settings()
    models_dir = settings.path("paths", "models_dir")
    metric_files = {
        "forecasting": models_dir / "forecasting" / "forecast_metrics.json",
        "churn": models_dir / "churn" / "churn_metrics.json",
    }
    for model_name, path in metric_files.items():
        if not path.exists():
            continue
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        for metric_name, value in payload.items():
            if isinstance(value, (int, float)):
                MODEL_METRIC_GAUGE.labels(model_name, metric_name).set(float(value))


class ChurnRequest(BaseModel):
    customer_id: str = Field(..., example="C12345")
    customer_lifetime_value: float = 1500.0
    avg_order_value: float = 75.0
    purchase_frequency: float = 0.05
    days_since_last_purchase: int = 45
    rolling_7day_sales: float = 12000.0
    rolling_30day_sales: float = 48000.0
    seasonal_index: float = 1.1
    inventory_risk_score: float = 0.3


class ForecastRequest(BaseModel):
    horizon_days: int = Field(30, ge=1, le=90)


def _load_churn_model():
    settings = get_settings()
    path = settings.path("paths", "models_dir") / "churn" / "churn_model.pkl"
    if not path.exists():
        raise HTTPException(503, "Churn model not trained. Run pipeline first.")
    return load_pickle(path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "RetailPulse"}


@app.get("/metrics")
def prometheus_metrics() -> Response:
    _refresh_model_metric_gauges()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/metrics")
def model_metrics() -> dict[str, Any]:
    settings = get_settings()
    models_dir = settings.path("paths", "models_dir")
    metrics = {}
    for name in ["forecasting/forecast_metrics.json", "churn/churn_metrics.json"]:
        path = models_dir / name
        if path.exists():
            import json
            with open(path) as fh:
                metrics[name.split("/")[0]] = json.load(fh)
    return metrics


@app.post("/api/v1/churn/predict")
def predict_churn(req: ChurnRequest) -> dict[str, Any]:
    model = _load_churn_model()
    features = pd.DataFrame([{
        "CustomerLifetimeValue": req.customer_lifetime_value,
        "AvgOrderValue": req.avg_order_value,
        "PurchaseFrequency": req.purchase_frequency,
        "DaysSinceLastPurchase": req.days_since_last_purchase,
        "Rolling7DaySales": req.rolling_7day_sales,
        "Rolling30DaySales": req.rolling_30day_sales,
        "SeasonalIndex": req.seasonal_index,
        "InventoryRiskScore": req.inventory_risk_score,
    }])
    proba = float(model.predict_proba(features)[0, 1])
    return {
        "customer_id": req.customer_id,
        "churn_probability": proba,
        "churn_risk": "High" if proba >= 0.7 else "Medium" if proba >= 0.4 else "Low",
    }


@app.post("/api/v1/forecast")
def get_forecast(req: ForecastRequest) -> dict[str, Any]:
    settings = get_settings()
    model_path = settings.path("paths", "models_dir") / "forecasting" / "forecasting_model.pkl"
    if not model_path.exists():
        raise HTTPException(503, "Forecast model not trained.")
    try:
        ingestion = DataIngestion()
        df = ingestion.load_processed()
    except Exception:
        df = DataIngestion().load()
    from src.models.forecasting import HybridDemandForecaster
    artifacts = load_pickle(model_path)
    forecaster = HybridDemandForecaster()
    forecaster.prophet_model = artifacts["prophet"]
    forecast = forecaster.forecast_future(df, horizon=req.horizon_days)
    return {
        "horizon_days": req.horizon_days,
        "forecast": forecast.to_dict(orient="records"),
    }


@app.get("/api/v1/inventory")
def inventory_recommendations(limit: int = 50) -> dict[str, Any]:
    settings = get_settings()
    path = settings.path("paths", "models_dir") / "inventory" / "reorder_recommendations.parquet"
    if not path.exists():
        raise HTTPException(503, "Inventory optimization not run.")
    df = pd.read_parquet(path).head(limit)
    return {"count": len(df), "recommendations": df.to_dict(orient="records")}


@app.get("/api/v1/segments")
def customer_segments(limit: int = 100) -> dict[str, Any]:
    settings = get_settings()
    path = settings.path("paths", "models_dir") / "segmentation" / "customer_segments.parquet"
    if not path.exists():
        raise HTTPException(503, "Segmentation not run.")
    df = pd.read_parquet(path).head(limit)
    return {"count": len(df), "segments": df.reset_index().to_dict(orient="records")}
