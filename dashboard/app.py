"""
RetailPulse Streamlit Dashboard
AI-Powered Customer Analytics & Demand Forecasting Platform
"""

import streamlit as st

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics & Demand Forecasting")
st.sidebar.markdown("---")
st.sidebar.info(
    "**Navigation:** Use the sidebar pages to explore analytics, "
    "forecasts, segmentation, churn, inventory, SHAP, and model metrics."
)

pages = {
    "Executive Overview": "pages/1_Executive_Overview.py",
    "Sales Analytics": "pages/2_Sales_Analytics.py",
    "Demand Forecasting": "pages/3_Demand_Forecasting.py",
    "Customer Segmentation": "pages/4_Customer_Segmentation.py",
    "Churn Prediction": "pages/5_Churn_Prediction.py",
    "Inventory Optimization": "pages/6_Inventory_Optimization.py",
    "SHAP Explainability": "pages/7_SHAP_Explainability.py",
    "Model Metrics": "pages/8_Model_Metrics.py",
}

st.title("Welcome to RetailPulse")
st.markdown(
    """
    End-to-end retail analytics platform delivering:
    - **Demand forecasting** (Prophet + XGBoost + LSTM hybrid)
    - **Customer segmentation** (KMeans + DBSCAN)
    - **Churn prediction** (XGBoost + Optuna)
    - **Inventory optimization** with safety stock
    - **SHAP explainability** and **Evidently drift** monitoring
    """
)
st.markdown("Select a page from the sidebar to begin.")
