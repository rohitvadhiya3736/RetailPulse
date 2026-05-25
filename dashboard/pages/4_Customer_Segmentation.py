"""Customer Segmentation dashboard page."""

import plotly.express as px
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_processed, load_segments

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation")
st.caption("KMeans + DBSCAN on RFM features")

segments = load_segments()
df = load_processed()

if not segments.empty:
    fig = px.scatter(
        segments.reset_index(),
        x="recency",
        y="monetary",
        color="SegmentLabel" if "SegmentLabel" in segments.columns else "KMeansSegment",
        title="RFM Segmentation Map",
        opacity=0.6,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(segments.head(100), use_container_width=True)
elif "CustomerSegment" in df.columns:
    seg = df.groupby("CustomerSegment")["Customer ID"].nunique().reset_index()
    st.plotly_chart(px.bar(seg, x="CustomerSegment", y="Customer ID", title="Customers per Segment"))
else:
    st.warning("Run training pipeline for segmentation artifacts.")
