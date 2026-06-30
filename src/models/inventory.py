#Inventory optimization using forecasted demand and safety stock.

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config.loader import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InventoryOptimizer:

    def __init__(self) -> None:
        self.settings = get_settings()

    def optimize(self, df: pd.DataFrame, forecast_df: pd.DataFrame | None = None) -> pd.DataFrame:
        lead_time = int(self.settings.get("inventory", "lead_time_days", default=7))
        z = float(self.settings.get("inventory", "safety_stock_z", default=1.65))
        multiplier = float(self.settings.get("inventory", "reorder_multiplier", default=1.2))

        sku = df.groupby("StockCode").agg(
            avg_daily_demand=("Quantity", "mean"),
            demand_std=("Quantity", "std"),
            current_velocity=("Quantity", "sum"),
            risk_score=("InventoryRiskScore", "mean"),
            category=("ProductCategory", "first"),
        )
        sku["demand_std"] = sku["demand_std"].fillna(sku["avg_daily_demand"] * 0.3)

        if forecast_df is not None and "yhat" in forecast_df.columns:
            forecast_daily = forecast_df["yhat"].mean() / max(len(df["StockCode"].unique()), 1)
            sku["forecast_demand"] = forecast_daily
        else:
            sku["forecast_demand"] = sku["avg_daily_demand"]

        sku["safety_stock"] = z * sku["demand_std"] * np.sqrt(lead_time)
        sku["reorder_point"] = sku["forecast_demand"] * lead_time + sku["safety_stock"]
        sku["economic_order_qty"] = np.sqrt(
            2 * sku["current_velocity"] * 50 / (sku["avg_daily_demand"].clip(lower=0.1) * 0.2)
        )
        sku["recommended_order_qty"] = (
            (sku["reorder_point"] + sku["economic_order_qty"]) * multiplier
        ).round(0)

        sku["stock_status"] = np.where(
            sku["risk_score"] > 0.7,
            "Critical – Reorder Now",
            np.where(sku["risk_score"] > 0.4, "Monitor", "Healthy"),
        )
        sku = sku.reset_index()
        logger.info("Inventory optimization complete for %d SKUs", len(sku))
        return sku
