
from __future__ import annotations

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config.loader import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftMonitor:
    NUMERIC_FEATURES = [
        "Quantity",
        "Price",
        "TotalAmount",
        "CustomerLifetimeValue",
        "PurchaseFrequency",
        "Rolling7DaySales",
    ]

    def __init__(self) -> None:
        self.settings = get_settings()

    def run_report(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        output_path: Path | None = None,
    ) -> dict:
        output_path = output_path or self.settings.project_root / self.settings.get(
            "evidently", "report_path", default="artifacts/reports/drift_report.html"
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cols = [c for c in self.NUMERIC_FEATURES if c in reference.columns and c in current.columns]
        ref = reference[cols].sample(min(5000, len(reference)), random_state=42)
        cur = current[cols].sample(min(5000, len(current)), random_state=42)

        report = Report([DataDriftPreset()])
        snapshot = report.run(cur, ref)
        snapshot.save_html(str(output_path))

        drift_share = self._extract_drift_share(snapshot)
        threshold = float(self.settings.get("evidently", "drift_threshold", default=0.15))
        alert = drift_share > threshold
        logger.info("Drift share: %.2f%% | Alert: %s", drift_share * 100, alert)

        return {
            "drift_share": drift_share,
            "alert": alert,
            "report_path": str(output_path),
        }

    @staticmethod
    def _extract_drift_share(snapshot) -> float:
        for metric in snapshot.dict().get("metrics", []):
            if "DriftedColumnsCount" in metric.get("metric_name", ""):
                share = metric.get("value", {}).get("share")
                if share is not None:
                    return float(share)
        return 0.0
