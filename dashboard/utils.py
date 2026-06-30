"""Shared dashboard data loaders with cloud-safe demo fallbacks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA = ROOT / "data/processed/retail_features.parquet"
RAW_DATA = ROOT / "data/raw/online_retail.csv"
FORCE_DEMO = os.getenv("RETAILPULSE_DEMO_MODE", "").lower() in {"1", "true", "yes"}


@st.cache_data(ttl=300)
def load_processed() -> pd.DataFrame:
    if PROCESSED_DATA.exists() and not FORCE_DEMO:
        return pd.read_parquet(PROCESSED_DATA)
    if RAW_DATA.exists() and not FORCE_DEMO:
        return pd.read_csv(RAW_DATA, parse_dates=["InvoiceDate"])
    return _build_demo_data()


def using_demo_data() -> bool:
    """Return whether the dashboard is using its deterministic public demo."""
    return FORCE_DEMO or (not PROCESSED_DATA.exists() and not RAW_DATA.exists())


def render_data_status() -> None:
    """Clearly identify synthetic data on public deployments."""
    if using_demo_data():
        st.caption(
            "Public demo mode: deterministic synthetic transactions are shown "
            "because production datasets and trained artifacts are not stored in Git."
        )


@st.cache_data
def _build_demo_data(rows: int = 8_000, seed: int = 42) -> pd.DataFrame:
    """Create a realistic, deterministic feature-store sample for the live app."""
    rng = np.random.default_rng(seed)
    customers = np.array([f"CUST{i:04d}" for i in range(1, 451)])
    countries = np.array(
        ["United Kingdom", "Germany", "France", "Netherlands", "Spain", "Australia"]
    )
    country_weights = np.array([0.58, 0.12, 0.11, 0.08, 0.07, 0.04])
    categories = np.array(
        ["Home", "Kitchen", "Gifts", "Office", "Seasonal", "Accessories"]
    )
    category_prices = np.array([22.0, 18.0, 14.0, 11.0, 16.0, 9.0])

    product_count = 120
    product_ids = np.arange(product_count)
    product_categories = categories[product_ids % len(categories)]
    product_names = np.array(
        [f"{product_categories[i]} Product {i + 1:03d}" for i in product_ids]
    )
    product_prices = category_prices[product_ids % len(categories)] * rng.lognormal(
        mean=0.0, sigma=0.28, size=product_count
    )

    product_index = rng.integers(0, product_count, size=rows)
    base_date = pd.Timestamp("2025-07-01")
    invoice_dates = (
        base_date
        + pd.to_timedelta(rng.integers(0, 365, size=rows), unit="D")
        + pd.to_timedelta(rng.integers(8, 21, size=rows), unit="h")
        + pd.to_timedelta(rng.integers(0, 60, size=rows), unit="m")
    )
    quantity = np.clip(rng.negative_binomial(3, 0.55, size=rows) + 1, 1, 30)
    price = product_prices[product_index] * rng.uniform(0.9, 1.1, size=rows)

    df = pd.DataFrame(
        {
            "Invoice": [f"INV{100_000 + i:06d}" for i in range(rows)],
            "StockCode": [f"SKU{i + 1:04d}" for i in product_index],
            "Description": product_names[product_index],
            "Quantity": quantity,
            "InvoiceDate": invoice_dates,
            "Price": np.round(price, 2),
            "Customer ID": rng.choice(customers, size=rows),
            "Country": rng.choice(countries, size=rows, p=country_weights),
            "ProductCategory": product_categories[product_index],
        }
    ).sort_values("InvoiceDate", ignore_index=True)

    df["TotalAmount"] = df["Quantity"] * df["Price"]
    df["InvoiceMonth"] = df["InvoiceDate"].dt.month
    df["InvoiceDay"] = df["InvoiceDate"].dt.day
    df["InvoiceHour"] = df["InvoiceDate"].dt.hour
    df["WeekendFlag"] = df["InvoiceDate"].dt.dayofweek.ge(5).astype(int)

    customer_group = df.groupby("Customer ID", observed=True)
    first_purchase = customer_group["InvoiceDate"].transform("min")
    last_purchase = customer_group["InvoiceDate"].transform("max")
    lifetime_days = (last_purchase - first_purchase).dt.days.add(1).clip(lower=1)
    df["CustomerLifetimeValue"] = customer_group["TotalAmount"].transform("sum")
    df["AvgOrderValue"] = customer_group["TotalAmount"].transform("mean")
    df["PurchaseFrequency"] = (
        customer_group["Invoice"].transform("nunique") / lifetime_days
    )
    reference_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    df["DaysSinceLastPurchase"] = (
        reference_date - last_purchase.dt.normalize()
    ).dt.days

    sales_date = df["InvoiceDate"].dt.normalize()
    daily_sales = df.groupby(sales_date, observed=True)["TotalAmount"].sum()
    full_dates = pd.date_range(daily_sales.index.min(), daily_sales.index.max(), freq="D")
    daily_sales = daily_sales.reindex(full_dates, fill_value=0.0)
    df["daily_sales"] = sales_date.map(daily_sales)
    df["Rolling7DaySales"] = sales_date.map(
        daily_sales.rolling(7, min_periods=1).sum()
    )
    df["Rolling30DaySales"] = sales_date.map(
        daily_sales.rolling(30, min_periods=1).sum()
    )
    df["Lag_1_Day_Sales"] = sales_date.map(daily_sales.shift(1).fillna(0.0))
    df["Lag_7_Day_Sales"] = sales_date.map(daily_sales.shift(7).fillna(0.0))

    country_revenue = df.groupby("Country", observed=True)["TotalAmount"].sum()
    country_rank = country_revenue.rank(method="dense", ascending=False).astype(int)
    df["CountrySalesRank"] = df["Country"].map(country_rank)

    customer_summary = (
        df.groupby("Customer ID", observed=True)
        .agg(
            monetary=("CustomerLifetimeValue", "first"),
            recency=("DaysSinceLastPurchase", "first"),
            frequency=("PurchaseFrequency", "first"),
        )
    )
    high_value = customer_summary["monetary"].quantile(0.70)
    low_value = customer_summary["monetary"].quantile(0.35)
    recent = customer_summary["recency"].quantile(0.35)
    inactive = customer_summary["recency"].quantile(0.75)
    frequent = customer_summary["frequency"].quantile(0.65)
    customer_summary["segment"] = np.select(
        [
            (customer_summary["monetary"] >= high_value)
            & (customer_summary["recency"] <= recent),
            customer_summary["frequency"] >= frequent,
            customer_summary["monetary"] >= high_value,
            customer_summary["recency"] >= inactive,
            customer_summary["monetary"] <= low_value,
        ],
        ["Champions", "Loyal", "High Value", "At Risk", "New"],
        default="Regular",
    )
    df["CustomerSegment"] = df["Customer ID"].map(customer_summary["segment"])
    df["ChurnFlag"] = df["DaysSinceLastPurchase"].ge(inactive).astype(int)

    monthly_revenue = df.groupby("InvoiceMonth", observed=True)["TotalAmount"].sum()
    seasonal_index = monthly_revenue / monthly_revenue.mean()
    df["SeasonalIndex"] = df["InvoiceMonth"].map(seasonal_index)

    sku_variability = (
        df.groupby("StockCode", observed=True)["Quantity"].std().fillna(0.0)
    )
    risk = (sku_variability - sku_variability.min()) / (
        sku_variability.max() - sku_variability.min() + 1e-9
    )
    df["InventoryRiskScore"] = df["StockCode"].map(risk).clip(0.0, 1.0)
    return df


@st.cache_data(ttl=300)
def load_metrics() -> dict:
    metrics = {}
    for name in ["forecasting/forecast_metrics.json", "churn/churn_metrics.json"]:
        p = ROOT / "artifacts/models" / name
        if p.exists() and not FORCE_DEMO:
            with open(p) as fh:
                metrics[name.split("/")[0]] = json.load(fh)
    if metrics:
        return metrics
    return {
        "forecasting": {"mape": 27.49158598639517},
        "churn": {
            "roc_auc": 0.9999940238210492,
            "precision": 0.9912,
            "precision_at_top_20pct": 1.0,
        },
    }


@st.cache_data(ttl=300)
def load_inventory() -> pd.DataFrame:
    path = ROOT / "artifacts/models/inventory/reorder_recommendations.parquet"
    if path.exists() and not FORCE_DEMO:
        return pd.read_parquet(path)

    df = load_processed()
    days = max((df["InvoiceDate"].max() - df["InvoiceDate"].min()).days, 1)
    inventory = (
        df.groupby(["StockCode", "ProductCategory"], observed=True)
        .agg(total_demand=("Quantity", "sum"), demand_std=("Quantity", "std"))
        .reset_index()
        .rename(columns={"ProductCategory": "category"})
    )
    inventory["demand_std"] = inventory["demand_std"].fillna(0.0)
    inventory["avg_daily_demand"] = inventory["total_demand"] / days
    inventory["reorder_point"] = (
        inventory["avg_daily_demand"] * 7
        + 1.65 * inventory["demand_std"] * np.sqrt(7)
    )
    inventory["recommended_order_qty"] = np.ceil(
        inventory["avg_daily_demand"] * 30 + inventory["reorder_point"]
    )
    high_risk = inventory["reorder_point"].quantile(0.75)
    low_risk = inventory["reorder_point"].quantile(0.35)
    inventory["stock_status"] = np.select(
        [
            inventory["reorder_point"] >= high_risk,
            inventory["reorder_point"] <= low_risk,
        ],
        ["Critical - Reorder Now", "Healthy"],
        default="Monitor",
    )
    return inventory


@st.cache_data(ttl=300)
def load_segments() -> pd.DataFrame:
    path = ROOT / "artifacts/models/segmentation/customer_segments.parquet"
    if path.exists() and not FORCE_DEMO:
        return pd.read_parquet(path)

    df = load_processed()
    segments = (
        df.groupby("Customer ID", observed=True)
        .agg(
            recency=("DaysSinceLastPurchase", "first"),
            monetary=("CustomerLifetimeValue", "first"),
            frequency=("PurchaseFrequency", "first"),
            SegmentLabel=("CustomerSegment", "first"),
        )
    )
    segment_codes = {name: index for index, name in enumerate(segments["SegmentLabel"].unique())}
    segments["KMeansSegment"] = segments["SegmentLabel"].map(segment_codes)
    return segments


@st.cache_data(ttl=300)
def load_shap_importance() -> dict[str, float]:
    """Load trained SHAP values or use the latest pipeline demo values."""
    importance_files = list(ROOT.glob("**/shap_importance.json"))
    if importance_files and not FORCE_DEMO:
        with open(importance_files[-1], encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "DaysSinceLastPurchase": 3.3299,
        "CustomerLifetimeValue": 0.1526,
        "Rolling7DaySales": 0.1513,
        "PurchaseFrequency": 0.1150,
        "Rolling30DaySales": 0.0715,
        "SeasonalIndex": 0.0396,
        "AvgOrderValue": 0.0337,
        "InventoryRiskScore": 0.0076,
    }
