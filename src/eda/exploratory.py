
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config.loader import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExploratoryAnalysis:

    #setup
    def __init__(self, output_dir: Path | None = None) -> None:
        settings = get_settings()
        self.output_dir = output_dir or settings.path("paths", "plots_dir") / "eda"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_full_eda(self, df: pd.DataFrame) -> dict:
        logger.info("Running advanced EDA")
        summary = self._summary_statistics(df)
        self._plot_sales_trend(df)
        self._plot_country_distribution(df)
        self._plot_customer_value(df)
        self._plot_seasonality(df)
        self._plot_correlation_heatmap(df)
        self._plot_outlier_analysis(df)
        return summary
    # quick overview before ml apply
    def _summary_statistics(self, df: pd.DataFrame) -> dict:
        summary = {
            "n_rows": len(df),
            "n_customers": df["Customer ID"].nunique(),
            "n_products": df["StockCode"].nunique(),
            "n_countries": df["Country"].nunique(),
            "date_range": [
                str(df["InvoiceDate"].min()),
                str(df["InvoiceDate"].max()),
            ],
            "total_revenue": float(df["TotalAmount"].sum()),
            "avg_order_value": float(df.groupby("Invoice")["TotalAmount"].sum().mean()),
            "churn_rate": float(df.groupby("Customer ID")["ChurnFlag"].first().mean()),
        }
        return summary

    def _plot_sales_trend(self, df: pd.DataFrame) -> None:
        daily = df.groupby(df["InvoiceDate"].dt.date)["TotalAmount"].sum().reset_index()
        fig = px.line(daily, x="InvoiceDate", y="TotalAmount", title="Daily Revenue Trend")
        fig.write_html(self.output_dir / "sales_trend.html")

    def _plot_country_distribution(self, df: pd.DataFrame) -> None:
        top = df.groupby("Country")["TotalAmount"].sum().nlargest(15).reset_index()
        fig = px.bar(top, x="Country", y="TotalAmount", title="Top 15 Countries by Revenue")
        fig.write_html(self.output_dir / "country_revenue.html")

    def _plot_customer_value(self, df: pd.DataFrame) -> None:
        clv = df.groupby("Customer ID")["CustomerLifetimeValue"].first()
        fig = px.histogram(clv, nbins=50, title="Customer Lifetime Value Distribution")
        fig.write_html(self.output_dir / "clv_distribution.html")

    def _plot_seasonality(self, df: pd.DataFrame) -> None:
        monthly = df.groupby("InvoiceMonth")["TotalAmount"].sum().reset_index()
        fig = px.bar(monthly, x="InvoiceMonth", y="TotalAmount", title="Monthly Seasonality")
        fig.write_html(self.output_dir / "seasonality.html")

    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        cols = [
            "Quantity",
            "Price",
            "TotalAmount",
            "CustomerLifetimeValue",
            "PurchaseFrequency",
            "Rolling7DaySales",
            "SeasonalIndex",
            "InventoryRiskScore",
        ]
        avail = [c for c in cols if c in df.columns]
        corr = df[avail].corr()
        fig = px.imshow(corr, text_auto=True, title="Feature Correlation Matrix")
        fig.write_html(self.output_dir / "correlation_heatmap.html")

    def _plot_outlier_analysis(self, df: pd.DataFrame) -> None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Price", "Quantity"))
        fig.add_trace(go.Box(y=df["Price"], name="Price"), row=1, col=1)
        fig.add_trace(go.Box(y=df["Quantity"], name="Quantity"), row=1, col=2)
        fig.update_layout(title="Outlier Detection – Box Plots")
        fig.write_html(self.output_dir / "outlier_boxplots.html")
