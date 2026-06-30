"""SHAP Explainability dashboard page."""

import streamlit as st
import sys
from pathlib import Path
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.utils import load_shap_importance, render_data_status

st.title("SHAP Explainability")
render_data_status()

root = Path(__file__).resolve().parents[2]
shap_dir = root / "artifacts/plots/shap"

col1, col2 = st.columns(2)
summary = shap_dir / "churn_shap_summary.png"
bar = shap_dir / "churn_shap_bar.png"
if summary.exists():
    col1.image(str(summary), caption="SHAP Summary Plot")
if bar.exists():
    col2.image(str(bar), caption="SHAP Feature Importance")

importance = load_shap_importance()
df_imp = {"feature": list(importance.keys()), "importance": list(importance.values())}
import pandas as pd
imp_df = pd.DataFrame(df_imp).sort_values("importance", ascending=True)
st.plotly_chart(
    px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
    ),
    width="stretch",
)
