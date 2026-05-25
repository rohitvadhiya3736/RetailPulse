
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config.loader import get_settings
from src.features.engineering import FeatureEngineer
from src.utils.io import save_pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerSegmentation:
    #RFM-based segmentation using KMeans and DBSCAN

    def __init__(self) -> None:
        self.settings = get_settings()
        self.scaler = StandardScaler()
        self.kmeans: KMeans | None = None
        self.dbscan_labels_: np.ndarray | None = None
        self.feature_engineer = FeatureEngineer()

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        rfm = self.feature_engineer.build_customer_rfm_matrix(df)
        X = self.scaler.fit_transform(rfm)

        n_clusters = int(self.settings.get("segmentation", "kmeans_clusters", default=6))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = self.kmeans.fit_predict(X)
        sil = silhouette_score(X, kmeans_labels)
        logger.info("KMeans silhouette score: %.4f", sil)

        eps = float(self.settings.get("segmentation", "dbscan_eps", default=0.8))
        min_samples = int(self.settings.get("segmentation", "dbscan_min_samples", default=5))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.dbscan_labels_ = dbscan.fit_predict(X)

        rfm_out = rfm.copy()
        rfm_out["KMeansSegment"] = kmeans_labels
        rfm_out["DBSCANSegment"] = self.dbscan_labels_
        rfm_out["SegmentLabel"] = rfm_out["KMeansSegment"].map(self._segment_names())
        return rfm_out

    def _segment_names(self) -> dict:
        return {
            0: "Budget Shoppers",
            1: "Loyal Regulars",
            2: "High Rollers",
            3: "Occasional Buyers",
            4: "New Prospects",
            5: "At-Risk Valuable",
        }

    def save_artifacts(self, rfm: pd.DataFrame, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_pickle({"scaler": self.scaler, "kmeans": self.kmeans}, output_dir / "segmentation_model.pkl")
        rfm.to_parquet(output_dir / "customer_segments.parquet")
        self._plot_segments(rfm, output_dir / "segmentation_chart.png")

    def _plot_segments(self, rfm: pd.DataFrame, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            rfm["recency"],
            rfm["monetary"],
            c=rfm["KMeansSegment"],
            cmap="viridis",
            alpha=0.6,
        )
        ax.set_xlabel("Recency (log)")
        ax.set_ylabel("Monetary (log)")
        ax.set_title("Customer Segmentation – KMeans")
        plt.colorbar(scatter, ax=ax)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
