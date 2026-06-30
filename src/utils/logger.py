"""Structured logging factory."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.config.loader import get_settings


def get_logger(name: str) -> logging.Logger:
    #Return a configured logger with console and optional file handler.
    settings = get_settings()
    level = getattr(logging, settings.get("logging", "level", default="INFO"))
    fmt = settings.get("logging", "format", default="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(fmt)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = settings.get("logging", "file")
    if log_file:
        path = settings.project_root / log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
