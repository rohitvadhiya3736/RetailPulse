#!/usr/bin/env python3
"""Train demand forecasting model only."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.loader import get_settings
from src.data.ingestion import DataIngestion
from src.models.forecasting import HybridDemandForecaster
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    df = DataIngestion().load_processed() if _processed_exists() else DataIngestion().load()
    forecaster = HybridDemandForecaster()
    metrics = forecaster.fit(df)
    forecaster.save_artifacts(get_settings().path("paths", "models_dir") / "forecasting")
    logger.info("Forecasting metrics: %s", metrics)


def _processed_exists() -> bool:
    return get_settings().path("paths", "processed_data").exists()


if __name__ == "__main__":
    main()
