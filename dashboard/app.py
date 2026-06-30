import plotly.io as pio
import streamlit as st

pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="RetailPulse",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page(
        "pages/1_Executive_Overview.py",
        title="Executive Overview",
        icon=":material/dashboard:",
        default=True,
    ),
    st.Page(
        "pages/2_Sales_Analytics.py",
        title="Sales Analytics",
        icon=":material/analytics:",
    ),
    st.Page(
        "pages/3_Demand_Forecasting.py",
        title="Demand Forecasting",
        icon=":material/trending_up:",
    ),
    st.Page(
        "pages/4_Customer_Segmentation.py",
        title="Customer Segmentation",
        icon=":material/groups:",
    ),
    st.Page(
        "pages/5_Churn_Prediction.py",
        title="Churn Prediction",
        icon=":material/person_alert:",
    ),
    st.Page(
        "pages/6_Inventory_Optimization.py",
        title="Inventory Optimization",
        icon=":material/inventory_2:",
    ),
    st.Page(
        "pages/7_SHAP_Explainability.py",
        title="SHAP Explainability",
        icon=":material/model_training:",
    ),
    st.Page(
        "pages/8_Model_Metrics.py",
        title="Model Metrics",
        icon=":material/speed:",
    ),
]

navigation = st.navigation(pages)
st.sidebar.title("RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics & Demand Forecasting")
navigation.run()
