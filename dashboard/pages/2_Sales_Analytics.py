"""Sales Analytics dashboard page."""

import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_processed

st.set_page_config(page_title="Sales Analytics", layout="wide")
st.title("Sales Analytics")

df = load_processed()
if df.empty:
    st.warning("No data available.")
    st.stop()

if "TotalAmount" not in df.columns:
    df["TotalAmount"] = df["Quantity"] * df["Price"]

tab1, tab2, tab3 = st.tabs(["Time Series", "Geography", "Products"])

with tab1:
    daily = df.groupby(df["InvoiceDate"].dt.date)["TotalAmount"].sum().reset_index()
    st.plotly_chart(px.line(daily, x="InvoiceDate", y="TotalAmount", title="Daily Sales"), use_container_width=True)
    if "InvoiceHour" in df.columns:
        hourly = df.groupby("InvoiceHour")["TotalAmount"].sum().reset_index()
        st.plotly_chart(px.bar(hourly, x="InvoiceHour", y="TotalAmount", title="Sales by Hour"), use_container_width=True)

with tab2:
    country = df.groupby("Country")["TotalAmount"].sum().nlargest(15).reset_index()
    st.plotly_chart(px.bar(country, x="Country", y="TotalAmount", title="Top Countries"), use_container_width=True)

with tab3:
    if "ProductCategory" in df.columns:
        cat = df.groupby("ProductCategory")["TotalAmount"].sum().reset_index()
        st.plotly_chart(px.treemap(cat, path=["ProductCategory"], values="TotalAmount"), use_container_width=True)
