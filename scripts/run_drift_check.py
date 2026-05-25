#!/usr/bin/env python3
"""Run Evidently drift detection."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.config.loader import get_settings
from src.monitoring.drift import DriftMonitor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    processed = settings.path("paths", "processed_data")
    reference = settings.path("paths", "reference_data")
    df = pd.read_parquet(processed)
    mid = len(df) // 2
    ref = pd.read_parquet(reference) if reference.exists() else df.iloc[:mid]
    cur = df.iloc[mid:]
    result = DriftMonitor().run_report(ref, cur)
    logger.info("Drift result: %s", result)


if __name__ == "__main__":
    main()
