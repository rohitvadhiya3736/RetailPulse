
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:
    #Production data cleaning pipeline.

    def __init__(self, price_cap_percentile: float = 99.5, qty_cap_percentile: float = 99.0) -> None:
        self.price_cap_percentile = price_cap_percentile
        self.qty_cap_percentile = qty_cap_percentile

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data cleaning on %d rows", len(df))
        out = df.copy()
        out = self._standardize_dtypes(out)
        out = self._handle_missing(out)
        out = self._remove_invalid_transactions(out)
        out = self._cap_outliers(out)
        out = self._deduplicate(out)
        logger.info("Cleaning complete: %d rows remain", len(out))
        return out

    def _standardize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Customer ID"] = df["Customer ID"].astype(str)
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        # Drop rows without critical fields
        critical = ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate"]
        before = len(df)
        df = df.dropna(subset=critical)
        logger.info("Dropped %d rows with missing critical fields", before - len(df))

        # Impute description from stock code mode
        if df["Description"].isna().any():
            mode_by_stock = df.groupby("StockCode")["Description"].apply(
                lambda s: s.mode().iloc[0] if len(s.mode()) else "Unknown"
            )
            df["Description"] = df["Description"].fillna(df["StockCode"].map(mode_by_stock))
            df["Description"] = df["Description"].fillna("Unknown Product")

        # Impute country with mode
        df["Country"] = df["Country"].fillna(df["Country"].mode().iloc[0])

        # Guest customers
        df["Customer ID"] = df["Customer ID"].replace({"nan": "GUEST", "None": "GUEST"})
        df["Customer ID"] = df["Customer ID"].fillna("GUEST")
        return df

    def _remove_invalid_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["Quantity"] > 0) & (df["Price"] > 0)
        removed = (~mask).sum()
        logger.info("Removed %d invalid quantity/price rows", removed)
        return df.loc[mask].copy()

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        price_cap = df["Price"].quantile(self.price_cap_percentile / 100)
        qty_cap = df["Quantity"].quantile(self.qty_cap_percentile / 100)
        df["Price"] = df["Price"].clip(upper=price_cap)
        df["Quantity"] = df["Quantity"].clip(upper=qty_cap)
        df["is_outlier_capped"] = (
            (df["Price"] >= price_cap) | (df["Quantity"] >= qty_cap)
        ).astype(int)
        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        subset = ["Invoice", "StockCode", "Quantity", "Price", "InvoiceDate", "Customer ID"]
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        logger.info("Removed %d duplicate rows", before - len(df))
        return df

    def detect_outliers_iqr(self, series: pd.Series, factor: float = 1.5) -> pd.Series:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        return ((series < lower) | (series > upper)).astype(int)
