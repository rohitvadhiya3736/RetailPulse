"""Churn Prediction dashboard page."""

import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_metrics, load_processed, render_data_status

st.title("Churn Prediction Analytics")
render_data_status()

df = load_processed()
metrics = load_metrics()

if metrics.get("churn"):
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{metrics['churn']['roc_auc']:.4f}")
    c2.metric("Precision", f"{metrics['churn']['precision']:.4f}")
    c3.metric("Precision @ Top 20%", f"{metrics['churn'].get('precision_at_top_20pct', 0):.4f}")

if df.empty or "ChurnFlag" not in df.columns:
    st.warning("Churn features not available.")
    st.stop()

cust = df.groupby("Customer ID").first().reset_index()
st.plotly_chart(
    px.histogram(cust, x="DaysSinceLastPurchase", color="ChurnFlag", barmode="overlay", title="Recency vs Churn"),
    width="stretch",
)
st.plotly_chart(
    px.scatter(cust, x="CustomerLifetimeValue", y="PurchaseFrequency", color="ChurnFlag", title="CLV vs Frequency"),
    width="stretch",
)

at_risk = cust.nlargest(20, "DaysSinceLastPurchase")[["Customer ID", "CustomerLifetimeValue", "DaysSinceLastPurchase", "ChurnFlag"]]
st.subheader("Top At-Risk Customers")
st.dataframe(at_risk, width="stretch")
