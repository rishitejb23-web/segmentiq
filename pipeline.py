# segmentiq/pipeline.py
"""
SegmentIQ - Behavioral Clustering and Propensity Modeling
----------------------------------------------------------
End-to-end customer analytics pipeline that:
  1. Segments customers into behaviorally distinct cohorts using
     K-Means and DBSCAN clustering on transactional data.
  2. Trains a Logistic Regression propensity model on each segment
     to score likelihood of product uptake / campaign response.
  3. Outputs scored customer records for targeted campaign activation.

Designed for retail banking use cases: credit card marketing,
deposit product cross-sell, and everyday banking engagement.

Author: Rishitej B
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    silhouette_score, davies_bouldin_score
)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ── 1. DATA LOADING & FEATURE ENGINEERING ────────────────────────────────────

def load_and_engineer(filepath: str) -> pd.DataFrame:
    """
    Load transactional customer data and engineer RFM + behavioural features.

    RFM (Recency, Frequency, Monetary) is the standard segmentation framework
    used in retail banking marketing analytics.

    Parameters
    ----------
    filepath : str
        Path to CSV with customer transaction records.
        Expected columns: customer_id, transaction_date, amount, channel,
                          product_type, balance, age, tenure_months

    Returns
    -------
    features : pd.DataFrame
        Customer-level feature matrix ready for clustering.
    """
    df = pd.read_csv(filepath, parse_dates=["transaction_date"])

    snapshot_date = df["transaction_date"].max()

    # ── RFM features ──
    rfm = df.groupby("customer_id").agg(
        recency   = ("transaction_date", lambda x: (snapshot_date - x.max()).days),
        frequency = ("transaction_date", "count"),
        monetary  = ("amount", "sum"),
    ).reset_index()

    # ── Behavioural features ──
    behavioural = df.groupby("customer_id").agg(
        avg_txn_amount  = ("amount", "mean"),
        std_txn_amount  = ("amount", "std"),
        digital_ratio   = ("channel", lambda x: (x == "digital").mean()),
        unique_products = ("product_type", "nunique"),
        avg_balance     = ("balance", "mean"),
    ).reset_index()

    # ── Demographics ──
    demographics = df.groupby("customer_id").agg(
        age            = ("age", "first"),
        tenure_months  = ("tenure_months", "first"),
    ).reset_index()

    # Merge all feature groups
    features = rfm.merge(behavioural, on="customer_id").merge(demographics, on="customer_id")

    # Fill std NaN (customers with single transaction)
    features["std_txn_amount"] = features["std_txn_amount"].fillna(0)

    return features


def generate_synthetic_data(n_customers: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer transaction data for demonstration.
    Produces realistic RFM distributions seen in retail banking datasets.

    Parameters
    ----------
    n_customers : int
        Number of unique customers to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Synthetic transaction records.
    """
    rng = np.random.default_rng(seed)
    n_txns = n_customers * 12  # ~12 transactions per customer per year

    customer_ids = rng.integers(1, n_customers + 1, size=n_txns)
    dates = pd.date_range("2022-01-01", "2023-12-31", periods=n_txns)

    # Segment-aware amount generation
    amounts = np.where(
        customer_ids % 3 == 0,
        rng.lognormal(mean=5.5, sigma=0.8, size=n_txns),   # high-value
        np.where(
            customer_ids % 3 == 1,
            rng.lognormal(mean=4.0, sigma=0.6, size=n_txns), # mid-value
            rng.lognormal(mean=3.0, sigma=0.5, size=n_txns)  # low-value
        )
    )

    df = pd.DataFrame({
        "customer_id":      customer_ids,
        "transaction_date": rng.choice(dates, size=n_txns),
        "amount":           amounts.round(2),
        "channel":          rng.choice(["digital", "branch", "atm"], size=n_txns, p=[0.6, 0.25, 0.15]),
        "product_type":     rng.choice(["checking", "savings", "credit_card", "loan"], size=n_txns),
        "balance":          rng.lognormal(mean=8, sigma=1.2, size=n_txns).round(2),
        "age":              rng.integers(22, 75, size=n_txns),
        "tenure_months":    rng.integers(1, 120, size=n_txns),
    })

    return df


# ── 2. CLUSTERING ────────────────────────────────────────────────────────────

def cluster_customers(
    features: pd.DataFrame,
    n_clusters: int = 4,
    method: str = "kmeans"
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Cluster customers using K-Means or DBSCAN.
    Evaluates cluster quality with Silhouette Score and Davies-Bouldin Index.

    Parameters
    ----------
    features : pd.DataFrame
        Customer feature matrix (excludes customer_id).
    n_clusters : int
        Number of clusters for K-Means (ignored for DBSCAN).
    method : str
        'kmeans' or 'dbscan'.

    Returns
    -------
    features_with_labels : pd.DataFrame
        Original features with cluster label appended.
    labels : np.ndarray
        Cluster assignment per customer.
    """
    feature_cols = [c for c in features.columns if c != "customer_id"]
    X = features[feature_cols].copy()

    # Scale — critical for distance-based clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nClustering {len(X):,} customers using {method.upper()}...")

    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(X_scaled)

    elif method == "dbscan":
        # eps and min_samples tuned for banking customer density
        clusterer = DBSCAN(eps=0.8, min_samples=50, n_jobs=-1)
        labels = clusterer.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise     = (labels == -1).sum()
        print(f"  DBSCAN found {n_clusters} clusters | Noise points: {n_noise:,}")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'dbscan'.")

    # Quality metrics (exclude noise points for DBSCAN)
    mask = labels != -1
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sil = silhouette_score(X_scaled[mask], labels[mask], sample_size=10_000)
        db  = davies_bouldin_score(X_scaled[mask], labels[mask])
        print(f"  Silhouette Score   : {sil:.4f}  (higher = better, range -1 to 1)")
        print(f"  Davies-Bouldin Idx : {db:.4f}   (lower = better)")

    features_with_labels = features.copy()
    features_with_labels["cluster"] = labels
    return features_with_labels, labels


def profile_clusters(features_with_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean feature values per cluster for business interpretation.
    This output is used to name and describe segments for marketing teams.

    Parameters
    ----------
    features_with_labels : pd.DataFrame
        Features with 'cluster' column.

    Returns
    -------
    profile : pd.DataFrame
        Cluster-level summary statistics.
    """
    feature_cols = [c for c in features_with_labels.columns
                    if c not in ["customer_id", "cluster"]]
    profile = features_with_labels.groupby("cluster")[feature_cols].mean().round(2)
    profile["n_customers"] = features_with_labels.groupby("cluster").size()

    # Assign descriptive names based on RFM profile
    # (These labels should be reviewed by the business / marketing team)
    def label_segment(row):
        if row["monetary"] > profile["monetary"].quantile(0.75):
            return "High Value"
        elif row["frequency"] > profile["frequency"].quantile(0.75):
            return "High Frequency"
        elif row["recency"] < profile["recency"].quantile(0.25):
            return "Recently Active"
        else:
            return "At Risk / Dormant"

    profile["segment_name"] = profile.apply(label_segment, axis=1)
    return profile


# ── 3. PCA VISUALISATION ─────────────────────────────────────────────────────

def plot_clusters(features_with_labels: pd.DataFrame) -> None:
    """
    Reduce to 2D with PCA and plot cluster assignments.
    Saved to outputs/cluster_plot.png.
    """
    feature_cols = [c for c in features_with_labels.columns
                    if c not in ["customer_id", "cluster"]]
    X_scaled = StandardScaler().fit_transform(features_with_labels[feature_cols])

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        coords[:, 0], coords[:, 1],
        c=features_with_labels["cluster"],
        cmap="tab10", alpha=0.4, s=5
    )
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("Customer Segments — PCA Projection")
    plt.tight_layout()
    plt.savefig("outputs/cluster_plot.png", dpi=150)
    plt.close()
    print("  Cluster plot saved to outputs/cluster_plot.png")


# ── 4. PROPENSITY MODELLING PER SEGMENT ──────────────────────────────────────

def train_propensity_models(
    features_with_labels: pd.DataFrame,
    target_col: str = "responded"
) -> dict:
    """
    Train a Logistic Regression propensity model for each customer segment.
    Predicts likelihood of responding to a targeted marketing campaign.

    In practice the target variable would come from a prior campaign's
    response data joined back to the customer feature table.

    Parameters
    ----------
    features_with_labels : pd.DataFrame
        Features + cluster labels + target variable.
    target_col : str
        Binary response variable (1 = responded, 0 = did not respond).

    Returns
    -------
    models : dict
        {cluster_id: fitted LogisticRegression model}
    """
    feature_cols = [c for c in features_with_labels.columns
                    if c not in ["customer_id", "cluster", target_col]]
    models = {}

    print(f"\nTraining per-segment propensity models (target: '{target_col}')...")

    for cluster_id in sorted(features_with_labels["cluster"].unique()):
        if cluster_id == -1:
            continue  # Skip DBSCAN noise points

        segment = features_with_labels[features_with_labels["cluster"] == cluster_id]
        if len(segment) < 100:
            print(f"  Segment {cluster_id}: too few records ({len(segment)}), skipping.")
            continue

        X = segment[feature_cols]
        y = segment[target_col]

        if y.nunique() < 2:
            print(f"  Segment {cluster_id}: single class in target, skipping.")
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=500, class_weight="balanced", random_state=42))
        ])

        # 5-fold CV AUC
        cv_aucs = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
        pipe.fit(X, y)
        models[cluster_id] = pipe

        print(f"  Segment {cluster_id} ({len(segment):,} customers) "
              f"| CV AUC: {cv_aucs.mean():.4f} +/- {cv_aucs.std():.4f}")

    return models


def score_customers(
    features_with_labels: pd.DataFrame,
    models: dict,
    target_col: str = "responded"
) -> pd.DataFrame:
    """
    Score all customers with their segment's propensity model.
    Output is used to rank customers for campaign prioritisation.

    Parameters
    ----------
    features_with_labels : pd.DataFrame
        Full customer feature table with cluster assignments.
    models : dict
        Per-segment propensity models from train_propensity_models().
    target_col : str
        Target column name (excluded from features).

    Returns
    -------
    scored : pd.DataFrame
        Customer IDs with propensity score and decile rank.
    """
    feature_cols = [c for c in features_with_labels.columns
                    if c not in ["customer_id", "cluster", target_col]]

    scored_chunks = []
    for cluster_id, model in models.items():
        segment = features_with_labels[features_with_labels["cluster"] == cluster_id].copy()
        segment["propensity_score"] = model.predict_proba(segment[feature_cols])[:, 1]
        scored_chunks.append(segment[["customer_id", "cluster", "propensity_score"]])

    scored = pd.concat(scored_chunks, ignore_index=True)

    # Decile ranking (10 = highest propensity)
    scored["decile"] = pd.qcut(scored["propensity_score"], q=10, labels=range(1, 11))

    scored.to_csv("outputs/scored_customers.csv", index=False)
    print(f"\n  Scored {len(scored):,} customers. Saved to outputs/scored_customers.csv")
    return scored


# ── 5. MAIN ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    # ── Generate or load data ──
    print("Generating synthetic customer transaction data (50,000 customers)...")
    raw_df = generate_synthetic_data(n_customers=50_000)

    # ── Feature engineering ──
    print("Engineering RFM and behavioural features...")
    features = load_and_engineer_from_df(raw_df)  # see note below
    print(f"  Feature matrix: {features.shape}")

    # ── K-Means clustering ──
    labeled, labels = cluster_customers(features, n_clusters=4, method="kmeans")

    # ── Cluster profiles ──
    profile = profile_clusters(labeled)
    print("\nCluster Profiles:")
    print(profile[["n_customers", "recency", "frequency", "monetary", "segment_name"]].to_string())
    profile.to_csv("outputs/cluster_profiles.csv")

    # ── Visualise ──
    plot_clusters(labeled)

    # ── Simulate response variable for propensity modelling ──
    # In production this comes from a prior campaign response join
    rng = np.random.default_rng(42)
    labeled["responded"] = rng.binomial(
        1,
        # High-value and recently-active customers respond more
        p=np.where(labeled["cluster"].isin([0, 2]), 0.25, 0.08),
        size=len(labeled)
    )

    # ── Train per-segment propensity models ──
    models = train_propensity_models(labeled, target_col="responded")

    # ── Score all customers ──
    scored = score_customers(labeled, models, target_col="responded")

    print("\nTop decile (highest propensity):")
    print(scored[scored["decile"] == 10].head(10).to_string(index=False))
    print("\nDone. Check the outputs/ folder.")


def load_and_engineer_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same as load_and_engineer() but accepts a DataFrame directly
    (used when data is generated in-memory rather than loaded from CSV).
    """
    snapshot_date = df["transaction_date"].max()

    rfm = df.groupby("customer_id").agg(
        recency   = ("transaction_date", lambda x: (snapshot_date - x.max()).days),
        frequency = ("transaction_date", "count"),
        monetary  = ("amount", "sum"),
    ).reset_index()

    behavioural = df.groupby("customer_id").agg(
        avg_txn_amount  = ("amount", "mean"),
        std_txn_amount  = ("amount", "std"),
        digital_ratio   = ("channel", lambda x: (x == "digital").mean()),
        unique_products = ("product_type", "nunique"),
        avg_balance     = ("balance", "mean"),
    ).reset_index()

    demographics = df.groupby("customer_id").agg(
        age           = ("age", "first"),
        tenure_months = ("tenure_months", "first"),
    ).reset_index()

    features = rfm.merge(behavioural, on="customer_id").merge(demographics, on="customer_id")
    features["std_txn_amount"] = features["std_txn_amount"].fillna(0)
    return features
