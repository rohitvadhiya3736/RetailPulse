#!/usr/bin/env python3
"""Train all RetailPulse models end-to-end."""

import os
import sys
from pathlib import Path

# Avoid OpenMP/threading crashes in CI and constrained environments
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from src.pipeline.orchestrator import RetailPulsePipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    results = RetailPulsePipeline().run()
    logger.info("Training complete: %s", results)


if __name__ == "__main__":
    main()
