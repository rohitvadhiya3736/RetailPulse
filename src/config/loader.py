"""YAML configuration loader with environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    """Typed application settings loaded from YAML."""

    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, *keys: str) -> Path:
        rel = self.get(*keys)
        if rel is None:
            raise KeyError(f"Missing path config: {'.'.join(keys)}")
        return (PROJECT_ROOT / rel).resolve()


def get_settings(config_path: Path | None = None) -> Settings:
    """Load settings from YAML; allow RETAILPULSE_CONFIG override."""
    path = config_path or Path(
        os.getenv("RETAILPULSE_CONFIG", PROJECT_ROOT / "src/config/settings.yaml")
    )
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Settings(raw=raw)
