"""Tests for raw data column alias normalization."""

import pandas as pd

from src.data.ingestion import REQUIRED_COLUMNS, DataIngestion, normalize_columns


def test_normalize_online_retail_aliases():
    df = pd.DataFrame(
        {
            "InvoiceNo": ["536365"],
            "StockCode": ["85123A"],
            "Description": ["WHITE HANGING HEART T-LIGHT HOLDER"],
            "Quantity": [6],
            "InvoiceDate": ["12/01/2010 08:26:00"],
            "UnitPrice": [2.55],
            "CustomerID": [17850],
            "Country": ["United Kingdom"],
        }
    )

    normalized = normalize_columns(df)

    assert list(normalized.columns) == REQUIRED_COLUMNS


def test_data_ingestion_accepts_online_retail_aliases(tmp_path):
    path = tmp_path / "online_retail.csv"
    pd.DataFrame(
        {
            "InvoiceNo": ["536365"],
            "StockCode": ["85123A"],
            "Description": ["WHITE HANGING HEART T-LIGHT HOLDER"],
            "Quantity": [6],
            "InvoiceDate": ["12/01/2010 08:26:00"],
            "UnitPrice": [2.55],
            "CustomerID": [17850],
            "Country": ["United Kingdom"],
        }
    ).to_csv(path, index=False)

    loaded = DataIngestion(source_path=path).load()

    assert list(loaded.columns) == REQUIRED_COLUMNS
    assert loaded.loc[0, "Invoice"] == 536365
    assert loaded.loc[0, "Price"] == 2.55
    assert loaded.loc[0, "Customer ID"] == 17850

