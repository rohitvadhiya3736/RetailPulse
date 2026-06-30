#!/usr/bin/env python3
"""Export RetailPulse outputs into Power BI-friendly CSV tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPORT_DIR = ROOT / "powerbi_exports"


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPORT_DIR / name
    df.to_csv(path, index=False)
    print(f"Exported {len(df):,} rows -> {path}")
    return path


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def export_transactions(features: pd.DataFrame) -> None:
    columns = [
        "Invoice",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "Price",
        "Customer ID",
        "Country",
        "TotalAmount",
        "InvoiceMonth",
        "InvoiceDay",
        "InvoiceHour",
        "WeekendFlag",
        "ProductCategory",
        "CustomerSegment",
        "ChurnFlag",
        "InventoryRiskScore",
    ]
    export = features[[col for col in columns if col in features.columns]].copy()
    _write_csv(export, "transactions.csv")


def export_daily_sales(features: pd.DataFrame) -> None:
    daily = (
        features.groupby(features["InvoiceDate"].dt.date)
        .agg(
            TotalSales=("TotalAmount", "sum"),
            TotalQuantity=("Quantity", "sum"),
            Orders=("Invoice", "nunique"),
            Customers=("Customer ID", "nunique"),
        )
        .reset_index()
        .rename(columns={"InvoiceDate": "SalesDate"})
    )
    daily["SalesDate"] = pd.to_datetime(daily["SalesDate"])
    daily["AOV"] = daily["TotalSales"] / daily["Orders"].clip(lower=1)
    _write_csv(daily, "daily_sales.csv")


def export_customer_summary(features: pd.DataFrame) -> None:
    snapshot = features["InvoiceDate"].max()
    customers = (
        features.groupby("Customer ID")
        .agg(
            CustomerLifetimeValue=("CustomerLifetimeValue", "first"),
            AvgOrderValue=("AvgOrderValue", "first"),
            PurchaseFrequency=("PurchaseFrequency", "first"),
            DaysSinceLastPurchase=("DaysSinceLastPurchase", "first"),
            Orders=("Invoice", "nunique"),
            Quantity=("Quantity", "sum"),
            Revenue=("TotalAmount", "sum"),
            LastPurchase=("InvoiceDate", "max"),
            Country=("Country", "first"),
            CustomerSegment=("CustomerSegment", "first"),
            ChurnFlag=("ChurnFlag", "first"),
        )
        .reset_index()
    )
    customers["SnapshotDate"] = snapshot
    _write_csv(customers, "customer_summary.csv")


def export_segments() -> None:
    path = ROOT / "artifacts/models/segmentation/customer_segments.parquet"
    if not path.exists():
        return
    segments = pd.read_parquet(path).reset_index()
    _write_csv(segments, "customer_segments.csv")


def export_inventory() -> None:
    path = ROOT / "artifacts/models/inventory/reorder_recommendations.parquet"
    if not path.exists():
        return
    inventory = pd.read_parquet(path)
    _write_csv(inventory, "inventory_recommendations.csv")


def export_model_metrics() -> None:
    forecast = _load_json(ROOT / "artifacts/models/forecasting/forecast_metrics.json")
    churn = _load_json(ROOT / "artifacts/models/churn/churn_metrics.json")
    rows = []
    if forecast:
        rows.extend(
            [
                {"Model": "Demand Forecasting", "Metric": "MAPE", "Value": forecast.get("mape")},
                {"Model": "Demand Forecasting", "Metric": "MAE", "Value": forecast.get("mae")},
                {"Model": "Demand Forecasting", "Metric": "Target MAPE", "Value": 12.0},
            ]
        )
    if churn:
        rows.extend(
            [
                {"Model": "Churn Prediction", "Metric": "ROC-AUC", "Value": churn.get("roc_auc")},
                {"Model": "Churn Prediction", "Metric": "Precision", "Value": churn.get("precision")},
                {
                    "Model": "Churn Prediction",
                    "Metric": "Precision@Top20%",
                    "Value": churn.get("precision_at_top_20pct"),
                },
                {"Model": "Churn Prediction", "Metric": "Target ROC-AUC", "Value": 0.88},
            ]
        )
    _write_csv(pd.DataFrame(rows), "model_metrics.csv")


def export_forecast(features: pd.DataFrame, horizon: int = 30) -> None:
    daily = (
        features.groupby(features["InvoiceDate"].dt.date)["TotalAmount"]
        .sum()
        .reset_index()
        .rename(columns={"InvoiceDate": "ds", "TotalAmount": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)

    try:
        from prophet import Prophet

        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model.fit(daily)
        future = model.make_future_dataframe(periods=horizon, freq="D")
        forecast = model.predict(future).tail(horizon)
        export = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(
            columns={
                "ds": "ForecastDate",
                "yhat": "ForecastSales",
                "yhat_lower": "ForecastLower",
                "yhat_upper": "ForecastUpper",
            }
        )
    except Exception:
        last_date = daily["ds"].max()
        baseline = daily["y"].tail(30).mean()
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")
        export = pd.DataFrame(
            {
                "ForecastDate": dates,
                "ForecastSales": baseline,
                "ForecastLower": baseline * 0.85,
                "ForecastUpper": baseline * 1.15,
            }
        )

    last_actual_date = features["InvoiceDate"].max().normalize()
    export["GeneratedFromDataThrough"] = last_actual_date
    _write_csv(export, "forecast_30_days.csv")


def export_dimension_tables(features: pd.DataFrame) -> None:
    _write_csv(
        features.groupby("Country", as_index=False)
        .agg(TotalSales=("TotalAmount", "sum"), Customers=("Customer ID", "nunique"))
        .sort_values("TotalSales", ascending=False),
        "country_sales.csv",
    )
    _write_csv(
        features.groupby("ProductCategory", as_index=False)
        .agg(TotalSales=("TotalAmount", "sum"), Quantity=("Quantity", "sum"), SKUs=("StockCode", "nunique"))
        .sort_values("TotalSales", ascending=False),
        "product_category_sales.csv",
    )


def main() -> None:
    features_path = ROOT / "data/processed/retail_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError("Run `python scripts/train_all.py` before exporting Power BI files.")

    features = pd.read_parquet(features_path)
    features["InvoiceDate"] = pd.to_datetime(features["InvoiceDate"])

    export_transactions(features)
    export_daily_sales(features)
    export_customer_summary(features)
    export_segments()
    export_inventory()
    export_model_metrics()
    export_forecast(features)
    export_dimension_tables(features)

    print(f"\nPower BI export complete. Import CSV files from: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
