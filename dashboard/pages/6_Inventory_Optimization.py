"""Inventory Optimization dashboard page."""

import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_inventory, render_data_status

st.title("Inventory Optimization")
render_data_status()

inv = load_inventory()
if inv.empty:
    st.warning("Run pipeline for inventory recommendations.")
    st.stop()

status_filter = st.multiselect(
    "Stock Status",
    options=inv["stock_status"].unique().tolist(),
    default=inv["stock_status"].unique().tolist(),
)
filtered = inv[inv["stock_status"].isin(status_filter)]

col1, col2, col3 = st.columns(3)
col1.metric("SKUs Monitored", len(inv))
col2.metric("Critical SKUs", inv["stock_status"].str.startswith("Critical").sum())
col3.metric("Avg Reorder Qty", f"{inv['recommended_order_qty'].mean():.0f}")

st.plotly_chart(
    px.scatter(filtered, x="avg_daily_demand", y="recommended_order_qty", color="stock_status", hover_data=["StockCode"]),
    width="stretch",
)
st.dataframe(
    filtered[["StockCode", "category", "recommended_order_qty", "reorder_point", "stock_status"]].head(50),
    width="stretch",
)
