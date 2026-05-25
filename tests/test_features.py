"""Tests for feature engineering."""

import pandas as pd
import pytest
from src.data.cleaning import DataCleaner
from src.features.engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Invoice": ["INV1", "INV2"],
        "StockCode": ["S1", "S2"],
        "Description": ["MUG RED", "GIFT BOX"],
        "Quantity": [2, 5],
        "InvoiceDate": pd.to_datetime(["2021-01-15 10:00", "2021-02-20 14:30"]),
        "Price": [10.0, 5.0],
        "Customer ID": ["C001", "C002"],
        "Country": ["United Kingdom", "Germany"],
    })


def test_cleaner_removes_invalid(sample_df):
    sample_df.loc[0, "Quantity"] = -1
    clean = DataCleaner().clean(sample_df)
    assert (clean["Quantity"] > 0).all()


def test_feature_engineering_columns(sample_df):
    clean = DataCleaner().clean(sample_df)
    featured = FeatureEngineer().transform(clean)
    expected = [
        "TotalAmount", "InvoiceMonth", "WeekendFlag",
        "CustomerLifetimeValue", "ChurnFlag", "ProductCategory",
    ]
    for col in expected:
        assert col in featured.columns
