"""SHAP Explainability dashboard page."""

import json
import streamlit as st
import sys
from pathlib import Path
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="SHAP Explainability", layout="wide")
st.title("SHAP Explainability")

root = Path(__file__).resolve().parents[2]
shap_dir = root / "artifacts/plots/shap"
importance_path = root / "mlflow/mlruns"

col1, col2 = st.columns(2)
summary = shap_dir / "churn_shap_summary.png"
bar = shap_dir / "churn_shap_bar.png"
if summary.exists():
    col1.image(str(summary), caption="SHAP Summary Plot")
if bar.exists():
    col2.image(str(bar), caption="SHAP Feature Importance")

# Load from mlflow artifacts if available
imp_files = list(root.glob("**/shap_importance.json"))
if imp_files:
    with open(imp_files[-1]) as fh:
        importance = json.load(fh)
    df_imp = {"feature": list(importance.keys()), "importance": list(importance.values())}
    import pandas as pd
    imp_df = pd.DataFrame(df_imp).sort_values("importance", ascending=True)
    st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h", title="Feature Importance"), use_container_width=True)
else:
    st.info("SHAP plots generated after running the full pipeline.")
