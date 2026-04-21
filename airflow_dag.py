# segmentiq/airflow_dag.py
"""
SegmentIQ — Airflow DAG for Weekly Automated Retraining and Scoring
--------------------------------------------------------------------
Runs every Monday at 6am. Pulls fresh customer transaction data from
the data warehouse, retrains cluster and propensity models, scores
all active customers, and writes results to the output table.

Designed for production ML operations in a retail banking environment.

DAG Tasks:
  1. extract_features    — pull and engineer features from SQL warehouse
  2. cluster_customers   — run K-Means clustering on latest data
  3. train_propensity    — retrain per-segment propensity models
  4. score_customers     — score all active customers
  5. write_scores        — write scored table back to data warehouse
  6. send_report         — email cluster profile summary to marketing team

Author: Rishitej B
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago

import pandas as pd
import numpy as np
import pickle
import os

# Default DAG arguments
default_args = {
    "owner":            "rishitejb",
    "depends_on_past":  False,
    "email":            ["rishitejb23@gmail.com"],
    "email_on_failure": True,
    "email_on_retry":   False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
}

dag = DAG(
    dag_id="segmentiq_weekly_retrain",
    description="Weekly customer segmentation and propensity model retraining",
    default_args=default_args,
    schedule_interval="0 6 * * 1",   # Every Monday at 06:00
    start_date=days_ago(1),
    catchup=False,
    tags=["ml", "segmentation", "propensity", "banking"],
)


# ── Task functions ────────────────────────────────────────────────────────────

def extract_features(**context):
    """
    Pull last 12 months of customer transactions from the data warehouse
    and engineer RFM + behavioural features.
    Pushes feature DataFrame to XCom for downstream tasks.

    In production this connects to Redshift/Snowflake/BigQuery via
    an Airflow connection. Here we use the synthetic data generator.
    """
    from pipeline import generate_synthetic_data, load_and_engineer_from_df

    print("Extracting features from data warehouse...")
    raw_df   = generate_synthetic_data(n_customers=50_000)
    features = load_and_engineer_from_df(raw_df)

    # Persist to a temp path (XCom size limit is ~48KB; DataFrames go to disk)
    tmp_path = "/tmp/segmentiq_features.parquet"
    features.to_parquet(tmp_path, index=False)
    context["ti"].xcom_push(key="features_path", value=tmp_path)
    print(f"  Features extracted: {features.shape} -> {tmp_path}")


def cluster_customers_task(**context):
    """
    Load features, run K-Means clustering, save labeled DataFrame and
    cluster profiles to disk.
    """
    from pipeline import cluster_customers, profile_clusters, plot_clusters

    features_path = context["ti"].xcom_pull(key="features_path", task_ids="extract_features")
    features = pd.read_parquet(features_path)

    labeled, labels = cluster_customers(features, n_clusters=4, method="kmeans")
    profile  = profile_clusters(labeled)

    os.makedirs("outputs", exist_ok=True)
    labeled_path  = "/tmp/segmentiq_labeled.parquet"
    profile_path  = "outputs/cluster_profiles.csv"

    labeled.to_parquet(labeled_path, index=False)
    profile.to_csv(profile_path)
    plot_clusters(labeled)

    context["ti"].xcom_push(key="labeled_path",  value=labeled_path)
    context["ti"].xcom_push(key="profile_path",  value=profile_path)
    print(f"  Clustering complete. {labeled['cluster'].nunique()} segments found.")


def train_propensity_task(**context):
    """
    Load labeled customer data, simulate campaign response target,
    train per-segment propensity models, and pickle them to disk.
    """
    from pipeline import train_propensity_models

    labeled_path = context["ti"].xcom_pull(key="labeled_path", task_ids="cluster_customers_task")
    labeled = pd.read_parquet(labeled_path)

    # In production: join to actual campaign response data from CRM
    rng = np.random.default_rng(42)
    labeled["responded"] = rng.binomial(
        1,
        p=np.where(labeled["cluster"].isin([0, 2]), 0.25, 0.08),
        size=len(labeled)
    )

    models = train_propensity_models(labeled, target_col="responded")

    models_path = "/tmp/segmentiq_models.pkl"
    with open(models_path, "wb") as f:
        pickle.dump(models, f)

    context["ti"].xcom_push(key="models_path",  value=models_path)
    context["ti"].xcom_push(key="labeled_path", value=labeled_path)
    print(f"  Trained {len(models)} propensity models.")


def score_customers_task(**context):
    """
    Score all active customers using their segment's propensity model.
    Write scored output to outputs/ and push path to XCom.
    """
    from pipeline import score_customers

    labeled_path = context["ti"].xcom_pull(key="labeled_path", task_ids="train_propensity_task")
    models_path  = context["ti"].xcom_pull(key="models_path",  task_ids="train_propensity_task")

    labeled = pd.read_parquet(labeled_path)
    with open(models_path, "rb") as f:
        models = pickle.load(f)

    scored = score_customers(labeled, models, target_col="responded")

    scored_path = "outputs/scored_customers.csv"
    scored.to_csv(scored_path, index=False)
    context["ti"].xcom_push(key="scored_path", value=scored_path)
    print(f"  Scored {len(scored):,} customers. Top decile: {(scored['decile']==10).sum():,}")


def write_scores_task(**context):
    """
    Write scored customer table back to the data warehouse.
    In production this uses a SQLAlchemy connection to Redshift/Snowflake.
    Here we just verify the output file exists and log a summary.
    """
    scored_path = context["ti"].xcom_pull(key="scored_path", task_ids="score_customers_task")
    scored = pd.read_csv(scored_path)

    print(f"  Writing {len(scored):,} scored records to data warehouse...")
    # Production: scored.to_sql("customer_propensity_scores", engine, if_exists="replace")
    print("  Write complete.")


# ── DAG task definitions ──────────────────────────────────────────────────────

t1 = PythonOperator(
    task_id="extract_features",
    python_callable=extract_features,
    dag=dag,
)

t2 = PythonOperator(
    task_id="cluster_customers_task",
    python_callable=cluster_customers_task,
    dag=dag,
)

t3 = PythonOperator(
    task_id="train_propensity_task",
    python_callable=train_propensity_task,
    dag=dag,
)

t4 = PythonOperator(
    task_id="score_customers_task",
    python_callable=score_customers_task,
    dag=dag,
)

t5 = PythonOperator(
    task_id="write_scores_task",
    python_callable=write_scores_task,
    dag=dag,
)

t6 = EmailOperator(
    task_id="send_report",
    to=["marketing-analytics@company.com"],
    subject="SegmentIQ Weekly Report — {{ ds }}",
    html_content="""
        <h3>SegmentIQ Weekly Segmentation Report</h3>
        <p>Clustering and propensity scoring completed for the week of {{ ds }}.</p>
        <p>Cluster profiles and scored customer file have been updated in the data warehouse.</p>
        <p>Check the <a href='http://airflow.internal/dags/segmentiq_weekly_retrain'>Airflow DAG</a> for run details.</p>
    """,
    dag=dag,
)

# ── Task dependencies ─────────────────────────────────────────────────────────
t1 >> t2 >> t3 >> t4 >> t5 >> t6
