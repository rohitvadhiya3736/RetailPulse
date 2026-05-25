"""Shared dashboard data loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]


@st.cache_data(ttl=300)
def load_processed() -> pd.DataFrame:
    path = ROOT / "data/processed/retail_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = ROOT / "data/raw/online_retail.csv"
    if raw.exists():
        return pd.read_csv(raw, parse_dates=["InvoiceDate"])
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_metrics() -> dict:
    metrics = {}
    for name in ["forecasting/forecast_metrics.json", "churn/churn_metrics.json"]:
        p = ROOT / "artifacts/models" / name
        if p.exists():
            with open(p) as fh:
                metrics[name.split("/")[0]] = json.load(fh)
    return metrics


@st.cache_data(ttl=300)
def load_inventory() -> pd.DataFrame:
    path = ROOT / "artifacts/models/inventory/reorder_recommendations.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


@st.cache_data(ttl=300)
def load_segments() -> pd.DataFrame:
    path = ROOT / "artifacts/models/segmentation/customer_segments.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()
