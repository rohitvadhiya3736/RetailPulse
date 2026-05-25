#!/usr/bin/env python3
"""Train churn prediction model only."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.loader import get_settings
from src.data.ingestion import DataIngestion
from src.models.churn import ChurnPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    path = get_settings().path("paths", "processed_data")
    df = DataIngestion().load_processed() if path.exists() else DataIngestion().load()
    churn = ChurnPredictor()
    metrics = churn.fit(df)
    churn.save_artifacts(get_settings().path("paths", "models_dir") / "churn")
    logger.info("Churn metrics: %s", metrics)


if __name__ == "__main__":
    main()
