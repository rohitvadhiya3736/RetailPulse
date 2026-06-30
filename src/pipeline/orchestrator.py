from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd

from src.config.loader import get_settings
from src.data.cleaning import DataCleaner
from src.data.ingestion import DataIngestion
from src.data.validation import DataValidator
from src.eda.exploratory import ExploratoryAnalysis
from src.explainability.shap_analysis import SHAPExplainer
from src.features.engineering import FeatureEngineer
from src.models.churn import ChurnPredictor
from src.models.forecasting import HybridDemandForecaster
from src.models.inventory import InventoryOptimizer
from src.models.segmentation import CustomerSegmentation
from src.monitoring.drift import DriftMonitor
from src.utils.io import save_parquet
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetailPulsePipeline:
    #training and evaluation pipeline.

    def __init__(self) -> None:
        self.settings = get_settings()
        mlflow.set_tracking_uri(self.settings.get("mlflow", "tracking_uri", default="mlflow/mlruns"))
        mlflow.set_experiment(self.settings.get("mlflow", "experiment_name", default="retailpulse-production"))

    def run(self, skip_eda: bool = False) -> dict:
        logger.info("Starting RetailPulse pipeline")
        ingestion = DataIngestion()
        validator = DataValidator()
        cleaner = DataCleaner()
        engineer = FeatureEngineer()

        raw = ingestion.load()
        validator.validate_raw(raw)
        clean = cleaner.clean(raw)
        featured = engineer.transform(clean)
        validator.validate_features(featured)

        processed_path = self.settings.path("paths", "processed_data")
        save_parquet(featured, processed_path)

        if not skip_eda:
            ExploratoryAnalysis().run_full_eda(featured)

        models_dir = self.settings.path("paths", "models_dir")

        with mlflow.start_run(run_name="retailpulse_full_pipeline"):
            # Segmentation
            seg = CustomerSegmentation()
            segments = seg.fit(featured)
            seg.save_artifacts(segments, models_dir / "segmentation")
            mlflow.log_param("kmeans_clusters", self.settings.get("segmentation", "kmeans_clusters"))

            # Forecasting
            forecaster = HybridDemandForecaster()
            forecast_metrics = forecaster.fit(featured)
            forecaster.save_artifacts(models_dir / "forecasting")
            forecast_future = forecaster.forecast_future(featured)
            mlflow.log_metric("forecast_mape", forecast_metrics["mape"])

            # Churn
            churn = ChurnPredictor()
            churn_metrics = churn.fit(featured)
            churn.save_artifacts(models_dir / "churn")
            mlflow.log_metric("churn_roc_auc", churn_metrics["roc_auc"])

            # SHAP
            shap_exp = SHAPExplainer()
            shap_result = shap_exp.explain_from_dataframe(featured, churn.model)
            mlflow.log_dict(shap_result["importance"], "shap_importance.json")

            # Inventory
            inventory = InventoryOptimizer()
            inv_df = inventory.optimize(featured, forecast_future)
            inv_dir = models_dir / "inventory"
            inv_dir.mkdir(parents=True, exist_ok=True)
            inv_df.to_parquet(inv_dir / "reorder_recommendations.parquet")

            # Drift (reference vs recent slice)
            ref_path = self.settings.path("paths", "reference_data")
            mid = len(featured) // 2
            if ref_path.exists():
                reference = pd.read_parquet(ref_path)
            else:
                reference = featured.iloc[:mid]
                save_parquet(reference, ref_path)
            current = featured.iloc[mid:]
            drift = DriftMonitor().run_report(reference, current)

            mlflow.log_artifacts(str(models_dir), artifact_path="models")

        results = {
            "forecast_mape": forecast_metrics["mape"],
            "churn_roc_auc": churn_metrics["roc_auc"],
            "drift": drift,
            "shap": shap_result,
            "processed_rows": len(featured),
        }
        logger.info("Pipeline complete: %s", results)
        return results
