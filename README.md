# SegmentIQ - Behavioral Clustering and Propensity Modeling

An end-to-end customer analytics pipeline for retail banking. Segments customers into behaviorally distinct cohorts using **K-Means** and **DBSCAN** clustering on transactional RFM data, then trains a **Logistic Regression** propensity model per segment to score campaign response likelihood. Includes an **Airflow DAG** for automated weekly retraining and scoring.

## Use Case

Retail banking teams use this pipeline to:
- Identify distinct customer behavioral segments (high-value, high-frequency, dormant, at-risk)
- Score each customer's likelihood of responding to a targeted campaign (credit card cross-sell, deposit product offers, everyday banking engagement)
- Rank customers by propensity decile for campaign prioritisation

## Tech Stack

Python, K-Means, DBSCAN, Logistic Regression, scikit-learn, pandas, NumPy, SQL, Apache Airflow, Matplotlib, Docker, Jupyter, Git

## Project Structure

```
segmentiq/
├── pipeline.py          # Core clustering + propensity training pipeline
├── airflow_dag.py       # Airflow DAG for weekly automated retraining
├── requirements.txt     # Dependencies
├── Dockerfile
├── notebooks/
│   └── eda.ipynb        # Segment exploration and visualisation
└── outputs/             # Cluster plots, profiles, scored customers
```

## Quickstart

```bash
pip install -r requirements.txt
python pipeline.py
```

**Outputs generated:**
- `outputs/cluster_profiles.csv` — mean feature values per segment with business labels
- `outputs/cluster_plot.png` — PCA scatter plot of customer segments
- `outputs/scored_customers.csv` — all customers scored with propensity score and decile rank

## Clustering Approach

| Method | When to use |
|--------|-------------|
| K-Means | When number of segments is known or prescribed by the business |
| DBSCAN | When discovering natural density-based groupings; handles outliers as noise |

Both methods are evaluated with **Silhouette Score** and **Davies-Bouldin Index** for cluster quality.

## Propensity Model

A **Logistic Regression** model is trained per segment (not globally) — this captures the fact that the drivers of campaign response differ significantly between a high-value, digitally-active customer vs. a dormant branch-only customer. Models are evaluated with 5-fold cross-validated AUC.

## Sample Output

```
Cluster Profiles:
         n_customers  recency  frequency  monetary  segment_name
cluster
0              12482     18.3       14.2   4821.50    High Value
1              11903     45.1        8.7   1203.20    High Frequency
2              13204     12.8       11.3   2104.80    Recently Active
3              12411     89.4        4.1    412.60    At Risk / Dormant

Per-segment propensity models:
  Segment 0 (12,482 customers) | CV AUC: 0.81 +/- 0.02
  Segment 1 (11,903 customers) | CV AUC: 0.76 +/- 0.03
  Segment 2 (13,204 customers) | CV AUC: 0.79 +/- 0.02
  Segment 3 (12,411 customers) | CV AUC: 0.71 +/- 0.04
```
