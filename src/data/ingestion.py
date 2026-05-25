"""Data ingestion from CSV and optional API sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.loader import get_settings
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country",
]

COLUMN_ALIASES = {
    "Invoice": [
        "InvoiceNo",
        "Invoice No",
        "InvoiceNumber",
        "Invoice Number",
        "OrderNo",
        "Order No",
        "OrderID",
        "Order ID",
    ],
    "StockCode": ["Stock Code", "SKU", "Sku", "ProductCode", "Product Code"],
    "Description": ["Product Description", "ProductName", "Product Name", "Item Description"],
    "Quantity": ["Qty", "Units", "UnitQuantity", "Unit Quantity"],
    "InvoiceDate": ["Invoice Date", "InvoiceDatetime", "OrderDate", "Order Date", "Date"],
    "Price": ["UnitPrice", "Unit Price", "UnitCost", "Unit Cost", "PriceEach"],
    "Customer ID": [
        "CustomerID",
        "Customer Id",
        "Customer_ID",
        "CustomerNo",
        "Customer No",
        "ClientID",
    ],
    "Country": ["Market", "Region", "Nation"],
}


def _column_key(column: str) -> str:
    """Normalize a column name for case/space-insensitive alias matching."""

    return "".join(character for character in str(column).lower() if character.isalnum())


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename common Online Retail column names to the project schema."""

    normalized = df.copy()
    normalized.columns = [str(column).strip().replace("\ufeff", "") for column in normalized.columns]
    existing_by_key = {_column_key(column): column for column in normalized.columns}
    rename_map: dict[str, str] = {}

    for target, aliases in COLUMN_ALIASES.items():
        if target in normalized.columns:
            continue
        for candidate in [target, *aliases]:
            source = existing_by_key.get(_column_key(candidate))
            if source:
                rename_map[source] = target
                break

    if rename_map:
        logger.info("Normalizing raw data columns: %s", rename_map)
        normalized = normalized.rename(columns=rename_map)
    return normalized


class DataIngestion:
    """Load raw retail transaction data."""

    def __init__(self, source_path: Path | None = None) -> None:
        settings = get_settings()
        self.source_path = source_path or settings.path("paths", "raw_data")

    def load(self) -> pd.DataFrame:
        """Load CSV with schema validation."""
        logger.info("Loading raw data from %s", self.source_path)
        if not self.source_path.exists():
            raise DataValidationError(
                f"Raw data not found at {self.source_path}. "
                "Run: python scripts/generate_sample_data.py"
            )
        df = normalize_columns(pd.read_csv(self.source_path))
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing columns after alias normalization: {missing}. "
                f"Available columns: {sorted(df.columns)}"
            )
        logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))
        return df[REQUIRED_COLUMNS].copy()

    def load_processed(self) -> pd.DataFrame:
        """Load engineered feature dataset."""
        settings = get_settings()
        path = settings.path("paths", "processed_data")
        if not path.exists():
            raise DataValidationError(f"Processed data not found: {path}")
        return pd.read_parquet(path)
