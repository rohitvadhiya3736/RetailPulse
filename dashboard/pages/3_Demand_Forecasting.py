"""Demand Forecasting dashboard page."""

import streamlit as st
import sys
from pathlib import Path
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_metrics, load_processed, render_data_status

st.title("Demand Forecasting")
st.caption("Hybrid ensemble: Prophet + XGBoost + LSTM")
render_data_status()

df = load_processed()
metrics = load_metrics()

if metrics.get("forecasting"):
    st.metric("Ensemble MAPE", f"{metrics['forecasting']['mape']:.2f}%")

plot_path = Path(__file__).resolve().parents[2] / "artifacts/plots/forecast_evaluation.png"
if plot_path.exists():
    st.image(str(plot_path), caption="Forecast vs Actual")

if not df.empty and "daily_sales" in df.columns:
    daily = df.groupby(df["InvoiceDate"].dt.date).agg(
        actual=("TotalAmount", "sum"),
        rolling_7=("Rolling7DaySales", "mean"),
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["InvoiceDate"], y=daily["actual"], name="Actual"))
    fig.add_trace(go.Scatter(x=daily["InvoiceDate"], y=daily["rolling_7"], name="7-Day Rolling"))
    fig.update_layout(title="Sales with Rolling Average")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Train models to generate forecast plots: `python scripts/train_all.py`")
