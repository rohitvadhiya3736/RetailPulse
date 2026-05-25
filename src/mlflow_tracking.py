"""MLflow helper utilities."""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from src.config.loader import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_client() -> MlflowClient:
    settings = get_settings()
    uri = settings.get("mlflow", "tracking_uri", default="mlflow/mlruns")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def register_model(run_id: str, artifact_path: str, model_name: str) -> None:
    client = get_client()
    source = f"runs:/{run_id}/{artifact_path}"
    client.create_registered_model(model_name)
    mv = mlflow.register_model(source, model_name)
    logger.info("Registered model %s version %s", model_name, mv.version)
