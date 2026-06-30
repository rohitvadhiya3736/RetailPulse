#SHAP explainability for churn and forecasting models.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.models.churn import ChurnPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SHAPExplainer:
    #Generate SHAP summary and dependence plots.

    def __init__(self, output_dir: Path | None = None) -> None:
        from src.config.loader import get_settings

        settings = get_settings()
        self.output_dir = output_dir or settings.path("paths", "plots_dir") / "shap"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def explain_churn(self, model, X: pd.DataFrame, max_samples: int = 500) -> dict:
        logger.info("Computing SHAP values for churn model")
        sample = X.sample(min(len(X), max_samples), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        summary_path = self.output_dir / "churn_shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
        plt.tight_layout()
        bar_path = self.output_dir / "churn_shap_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()

        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = dict(zip(sample.columns, mean_abs.tolist()))
        return {"importance": importance, "summary_plot": str(summary_path), "bar_plot": str(bar_path)}

    def explain_from_dataframe(self, df: pd.DataFrame, model) -> dict:
        features = [f for f in ChurnPredictor.FEATURES if f in df.columns]
        cust = df.groupby("Customer ID").first().reset_index()
        X = cust[features].fillna(0)
        return self.explain_churn(model, X)
