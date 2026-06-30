"""Executive Overview dashboard page."""

import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_metrics, load_processed, render_data_status

st.title("Executive Overview")
render_data_status()

df = load_processed()
metrics = load_metrics()

if df.empty:
    st.warning("No data found. Run: `python scripts/generate_sample_data.py && python scripts/train_all.py`")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
revenue = df["TotalAmount"].sum() if "TotalAmount" in df.columns else (df["Quantity"] * df["Price"]).sum()
col1.metric("Total Revenue", f"${revenue:,.0f}")
col2.metric("Customers", f"{df['Customer ID'].nunique():,}")
col3.metric("Transactions", f"{len(df):,}")
churn_rate = df.groupby("Customer ID")["ChurnFlag"].first().mean() if "ChurnFlag" in df.columns else 0
col4.metric("Churn Rate", f"{churn_rate:.1%}")

if metrics:
    fc = metrics.get("forecasting", {})
    ch = metrics.get("churn", {})
    st.subheader("Model Performance")
    m1, m2 = st.columns(2)
    forecast_mape = fc.get("mape", 0)
    churn_auc = ch.get("roc_auc", 0)
    m1.metric(
        "Forecast MAPE",
        f"{forecast_mape:.2f}%",
        delta=f"{forecast_mape - 12:.2f} pp vs target",
        delta_color="inverse",
    )
    m2.metric(
        "Churn ROC-AUC",
        f"{churn_auc:.4f}",
        delta=f"{churn_auc - 0.88:.4f} vs target",
    )

daily = df.groupby(df["InvoiceDate"].dt.date)["TotalAmount"].sum().reset_index() if "TotalAmount" in df.columns else None
if daily is not None:
    fig = px.area(daily, x="InvoiceDate", y="TotalAmount", title="Revenue Trend")
    st.plotly_chart(fig, width="stretch")

if "CustomerSegment" in df.columns:
    seg = df.groupby("CustomerSegment")["TotalAmount"].sum().reset_index()
    fig2 = px.pie(seg, names="CustomerSegment", values="TotalAmount", title="Revenue by Segment")
    st.plotly_chart(fig2, width="stretch")
