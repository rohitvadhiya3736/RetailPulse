"""Data quality validation gates."""

from __future__ import annotations

import pandas as pd

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate datasets before modeling."""

    def validate_raw(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise DataValidationError("Dataset is empty")
        if df["InvoiceDate"].isna().all():
            raise DataValidationError("All InvoiceDate values are null")
        null_pct = df.isnull().mean().max()
        if null_pct > 0.5:
            raise DataValidationError(f"Column null rate exceeds 50%: {null_pct:.2%}")
        logger.info("Raw data validation passed")

    def validate_features(self, df: pd.DataFrame) -> None:
        required = [
            "TotalAmount",
            "CustomerLifetimeValue",
            "Rolling7DaySales",
            "CustomerSegment",
            "ChurnFlag",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing engineered features: {missing}")
        if df["TotalAmount"].isna().sum() > len(df) * 0.1:
            raise DataValidationError("TotalAmount has excessive nulls")
        logger.info("Feature validation passed")
