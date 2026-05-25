from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.loader import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.churn_days = int(self.settings.get("data", "churn_threshold_days", default=90))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering features for %d transactions", len(df))
        out = df.copy()
        out = self._base_features(out)
        out = self._customer_features(out)
        out = self._time_series_features(out)
        out = self._product_features(out)
        out = self._geo_features(out)
        out = self._segment_and_churn(out)
        out = self._seasonal_inventory(out)
        return out

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["TotalAmount"] = df["Quantity"] * df["Price"]
        df["InvoiceMonth"] = df["InvoiceDate"].dt.month
        df["InvoiceDay"] = df["InvoiceDate"].dt.day
        df["InvoiceHour"] = df["InvoiceDate"].dt.hour
        df["WeekendFlag"] = (df["InvoiceDate"].dt.dayofweek >= 5).astype(int)
        return df

    def _customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cust = df.groupby("Customer ID").agg(
            CustomerLifetimeValue=("TotalAmount", "sum"),
            order_count=("Invoice", "nunique"),
            last_purchase=("InvoiceDate", "max"),
            first_purchase=("InvoiceDate", "min"),
        )
        cust["AvgOrderValue"] = cust["CustomerLifetimeValue"] / cust["order_count"].clip(lower=1)
        span_days = (cust["last_purchase"] - cust["first_purchase"]).dt.days.clip(lower=1)
        cust["PurchaseFrequency"] = cust["order_count"] / span_days
        snapshot = df["InvoiceDate"].max()
        cust["DaysSinceLastPurchase"] = (snapshot - cust["last_purchase"]).dt.days

        df = df.merge(
            cust[
                [
                    "CustomerLifetimeValue",
                    "AvgOrderValue",
                    "PurchaseFrequency",
                    "DaysSinceLastPurchase",
                ]
            ],
            left_on="Customer ID",
            right_index=True,
            how="left",
        )
        return df

    def _time_series_features(self, df: pd.DataFrame) -> pd.DataFrame:
        daily = (
            df.groupby(df["InvoiceDate"].dt.date)["TotalAmount"]
            .sum()
            .reset_index()
            .rename(columns={"InvoiceDate": "date", "TotalAmount": "daily_sales"})
        )
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        daily["Rolling7DaySales"] = daily["daily_sales"].rolling(7, min_periods=1).mean()
        daily["Rolling30DaySales"] = daily["daily_sales"].rolling(30, min_periods=1).mean()
        daily["Lag_1_Day_Sales"] = daily["daily_sales"].shift(1)
        daily["Lag_7_Day_Sales"] = daily["daily_sales"].shift(7)
        daily["Lag_1_Day_Sales"] = daily["Lag_1_Day_Sales"].fillna(daily["daily_sales"].mean())
        daily["Lag_7_Day_Sales"] = daily["Lag_7_Day_Sales"].fillna(daily["daily_sales"].mean())

        df["sales_date"] = df["InvoiceDate"].dt.normalize()
        daily_renamed = daily.rename(columns={"date": "sales_date"})
        merge_cols = [
            "sales_date",
            "Rolling7DaySales",
            "Rolling30DaySales",
            "Lag_1_Day_Sales",
            "Lag_7_Day_Sales",
            "daily_sales",
        ]
        df = df.merge(daily_renamed[merge_cols], on="sales_date", how="left")
        return df

    def _product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        n_cat = int(self.settings.get("feature_engineering", "product_categories", default=8))
        # Category from description keywords + clustering proxy
        keywords = {
            "HOME": ["HOME", "BOX", "BASKET", "CANDLE", "FRAME"],
            "FASHION": ["BAG", "SHIRT", "DRESS", "SCARF", "SHOE"],
            "GIFT": ["GIFT", "CARD", "WRAP", "PARTY"],
            "KITCHEN": ["MUG", "CUP", "PLATE", "KITCHEN", "LUNCH"],
            "DECOR": ["LIGHT", "LAMP", "DECOR", "ORNAMENT"],
            "STATIONERY": ["PAPER", "PEN", "NOTEBOOK", "CARD"],
            "TOY": ["TOY", "GAME", "DOLL"],
            "OTHER": [],
        }

        def assign_category(desc: str) -> str:
            desc_u = str(desc).upper()
            for cat, kws in keywords.items():
                if cat == "OTHER":
                    continue
                if any(k in desc_u for k in kws):
                    return cat
            # Hash-based bucket for remaining SKUs
            return f"CAT_{hash(desc_u) % max(n_cat - len(keywords) + 1, 1)}"

        df["ProductCategory"] = df["Description"].apply(assign_category)
        return df

    def _geo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        country_sales = df.groupby("Country")["TotalAmount"].sum().rank(ascending=False)
        df["CountrySalesRank"] = df["Country"].map(country_sales)
        return df

    def _segment_and_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        # RFM-based segment labels (refined by KMeans in modeling)
        r = df.groupby("Customer ID")["DaysSinceLastPurchase"].transform("first")
        f = df.groupby("Customer ID")["Invoice"].transform("nunique")
        m = df.groupby("Customer ID")["CustomerLifetimeValue"].transform("first")

        r_score = pd.qcut(r.rank(method="first"), 4, labels=[4, 3, 2, 1], duplicates="drop")
        f_score = pd.qcut(f.rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop")
        m_score = pd.qcut(m.rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop")
        rfm = r_score.astype(int) + f_score.astype(int) + m_score.astype(int)

        def rfm_label(score: int) -> str:
            if score >= 10:
                return "Champions"
            if score >= 8:
                return "Loyal"
            if score >= 6:
                return "Potential"
            if score >= 4:
                return "At Risk"
            return "Hibernating"

        def _segment_for_group(g: pd.DataFrame) -> str:
            return rfm_label(int(rfm.loc[g.index[0]]))

        seg_map = df.groupby("Customer ID", group_keys=False).apply(
            _segment_for_group, include_groups=False
        )
        df["CustomerSegment"] = df["Customer ID"].map(seg_map)

        # Churn: no purchase in threshold days
        df["ChurnFlag"] = (df["DaysSinceLastPurchase"] > self.churn_days).astype(int)
        return df

    def _seasonal_inventory(self, df: pd.DataFrame) -> pd.DataFrame:
        monthly_avg = df.groupby("InvoiceMonth")["TotalAmount"].transform("mean")
        global_avg = df["TotalAmount"].mean()
        df["SeasonalIndex"] = monthly_avg / max(global_avg, 1e-6)

        # Inventory risk: high volatility SKU + low recent sales
        sku_stats = df.groupby("StockCode").agg(
            sales_std=("TotalAmount", "std"),
            sales_mean=("TotalAmount", "mean"),
            qty_sum=("Quantity", "sum"),
        )
        sku_stats["sales_std"] = sku_stats["sales_std"].fillna(0)
        sku_stats["cv"] = sku_stats["sales_std"] / sku_stats["sales_mean"].clip(lower=1e-6)
        sku_stats["InventoryRiskScore"] = (
            sku_stats["cv"].rank(pct=True) * 0.6
            + (1 - sku_stats["qty_sum"].rank(pct=True)) * 0.4
        ).clip(0, 1)
        df = df.merge(
            sku_stats[["InventoryRiskScore"]],
            left_on="StockCode",
            right_index=True,
            how="left",
        )
        df["InventoryRiskScore"] = df["InventoryRiskScore"].fillna(0.5)
        return df

    def build_customer_rfm_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        snapshot = df["InvoiceDate"].max()
        rfm = df.groupby("Customer ID").agg(
            recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            frequency=("Invoice", "nunique"),
            monetary=("TotalAmount", "sum"),
        )
        for col in rfm.columns:
            rfm[col] = np.log1p(rfm[col])
        return rfm
