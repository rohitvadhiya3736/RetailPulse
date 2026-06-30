"""
RetailPulse Airflow DAG
Daily batch pipeline for ETL, feature engineering, model training, and drift detection.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.bash import *
from airflow.providers.standard.operators.python import PythonOperator

from airflow import DAG
import airflow.operators.bash
import airflow.operators.python

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def run_pipeline(**context):
    from src.pipeline.orchestrator import RetailPulsePipeline
    results = RetailPulsePipeline().run(skip_eda=False)
    context["ti"].xcom_push(key="pipeline_results", value=results)
    return results


def run_drift(**context):
    from scripts.run_drift_check import main
    main()


default_args = {
    "owner": "retailpulse",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retailpulse_daily_pipeline",
    default_args=default_args,
    description="RetailPulse ETL + ML training + drift monitoring",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["retailpulse", "mlops", "forecasting"],
) as dag:

    generate_data = BashOperator(
        task_id="generate_sample_data_if_missing",
        bash_command=f"cd {PROJECT_ROOT} && python scripts/generate_sample_data.py",
    )

    train_pipeline = PythonOperator(
        task_id="run_ml_pipeline",
        python_callable=run_pipeline,
    )

    drift_check = PythonOperator(
        task_id="evidently_drift_detection",
        python_callable=run_drift,
    )

    notify_success = BashOperator(
        task_id="pipeline_complete",
        bash_command='echo "RetailPulse pipeline completed successfully"',
    )

    generate_data >> train_pipeline >> drift_check >> notify_success
