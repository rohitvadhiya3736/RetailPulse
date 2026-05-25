"""Model Metrics dashboard page."""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_metrics

st.set_page_config(page_title="Model Metrics", layout="wide")
st.title("Model Metrics & MLOps")

metrics = load_metrics()
root = Path(__file__).resolve().parents[2]

st.subheader("Production Targets")
t1, t2 = st.columns(2)
t1.success("Forecast MAPE Target: ≤ 12%")
t2.success("Churn ROC-AUC Target: ≥ 0.88")

if metrics:
    st.json(metrics)
else:
    st.warning("No metrics found. Run `python scripts/train_all.py`")

drift_report = root / "artifacts/reports/drift_report.html"
if drift_report.exists():
    st.subheader("Evidently Drift Report")
    with open(drift_report) as fh:
        st.components.v1.html(fh.read(), height=600, scrolling=True)
else:
    st.info("Drift report available after pipeline run.")

st.subheader("MLflow")
st.markdown(f"Tracking URI: `{root / 'mlflow/mlruns'}`")
st.markdown("Launch UI: `mlflow ui --backend-store-uri mlflow/mlruns`")
