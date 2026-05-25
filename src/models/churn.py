#Churn prediction with XGBoost and Optuna hyperparameter tuning.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score, classification_report
from sklearn.model_selection import train_test_split

from src.config.loader import get_settings
from src.utils.exceptions import ModelTrainingError
from src.utils.io import save_json, save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChurnPredictor:
    FEATURES = [
        "CustomerLifetimeValue",
        "AvgOrderValue",
        "PurchaseFrequency",
        "DaysSinceLastPurchase",
        "Rolling7DaySales",
        "Rolling30DaySales",
        "SeasonalIndex",
        "InventoryRiskScore",
    ]

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model: xgb.XGBClassifier | None = None
        self.metrics: dict = {}

    def _build_customer_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("Customer ID").first().reset_index()

    def fit(self, df: pd.DataFrame, tune: bool = True) -> dict:
        cust = self._build_customer_dataset(df)
        features = [f for f in self.FEATURES if f in cust.columns]
        X = cust[features].fillna(0)
        y = cust["ChurnFlag"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if tune:
            best_params = self._optuna_tune(X_train, y_train)
            self.model = xgb.XGBClassifier(**best_params, eval_metric="auc", random_state=42)
            self.model.fit(X_train, y_train)
        else:
            cfg = self.settings.get("churn", "xgboost", default={})
            self.model = xgb.XGBClassifier(
                n_estimators=cfg.get("n_estimators", 400),
                max_depth=cfg.get("max_depth", 5),
                learning_rate=cfg.get("learning_rate", 0.03),
                scale_pos_weight=cfg.get("scale_pos_weight", 2.5),
                eval_metric="auc",
                random_state=42,
            )
            self.model.fit(X_train, y_train)

        proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        preds = (proba >= 0.5).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)

        # Precision at top 20%
        k = max(1, int(len(proba) * 0.2))
        top_idx = np.argsort(proba)[-k:]
        precision_top20 = y_test.iloc[top_idx].mean()

        self.metrics = {
            "roc_auc": auc,
            "precision": precision,
            "precision_at_top_20pct": float(precision_top20),
            "report": classification_report(y_test, preds, output_dict=True),
        }

        target_auc = float(self.settings.get("churn", "target_auc", default=0.88))
        logger.info("Churn ROC-AUC: %.4f (target ≥ %.2f)", auc, target_auc)
        if auc < target_auc * 0.95:
            logger.warning(
                "Churn AUC %.4f below target %.2f – retrain with more data or tune hyperparameters",
                auc,
                target_auc,
            )

        return self.metrics

    def _optuna_tune(self, X: pd.DataFrame, y: pd.Series) -> dict:
        trials = int(self.settings.get("churn", "optuna_trials", default=30))

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
            }
            model = xgb.XGBClassifier(**params, eval_metric="auc", random_state=42)
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        logger.info("Optuna best AUC: %.4f", study.best_value)
        return study.best_params

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        cust = self._build_customer_dataset(df)
        features = [f for f in self.FEATURES if f in cust.columns]
        cust["churn_probability"] = self.model.predict_proba(cust[features].fillna(0))[:, 1]
        return cust[["Customer ID", "churn_probability", "ChurnFlag"]]

    def save_artifacts(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_pickle(self.model, output_dir / "churn_model.pkl")
        save_json(self.metrics, output_dir / "churn_metrics.json")
