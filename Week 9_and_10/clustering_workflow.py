"""
=============================================================================
Scikit-Learn Clustering Workflow (Pipelines)
Dataset: Customer Segmentation for Discount Targeting
Algorithms: K-Means, Hierarchical (Agglomerative), DBSCAN
=============================================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = "4"
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# Set styling for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

OUTPUT_DIR = "output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. DATA LOADING & INSPECTION
# -----------------------------------------------------------------------------
print("=" * 80)
print("1. LOADING & PREVIEWING DATASET")
print("=" * 80)

DATA_PATH = os.path.join("archive (3)", "customer_segmentation_discount.csv")
df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset successfully! Dimensions: {df.shape[0]} rows, {df.shape[1]} columns.\n")

print("Dataset Summary:")
print(df.info())

# Separate ID and target
id_col = 'user_id'
target_col = 'should_receive_discount'

feature_cols = [c for c in df.columns if c not in [id_col, target_col]]

num_cols = [
    'age', 'annual_income', 'current_session_duration_sec', 'time_spent_deviation',
    'pages_viewed', 'cart_items_count_this_session', 'cart_total_value',
    'lifetime_purchases_count', 'customer_conversion_rate', 'days_since_last_discount',
    'social_channels_followed_count'
]
cat_cols = ['traffic_source', 'product_category', 'previous_review_sentiment']
bin_cols = [
    'gender', 'device_mobile', 'basket_icon_click', 'scrolled_to_reviews',
    'checked_delivery_pdp', 'saw_checkout', 'sign_in', 'is_returning_user'
]

print(f"\nFeature breakdown:")
print(f" - Continuous Numerical ({len(num_cols)}): {num_cols}")
print(f" - Categorical ({len(cat_cols)}): {cat_cols}")
print(f" - Binary ({len(bin_cols)}): {bin_cols}")

# -----------------------------------------------------------------------------
# 2. SCIKIT-LEARN PIPELINE ARCHITECTURE (PREPROCESSING)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("2. BUILDING SCIKIT-LEARN PREPROCESSING PIPELINE")
print("=" * 80)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
        ('bin', 'passthrough', bin_cols)
    ],
    remainder='drop'
)

# Extract processed feature names
preprocessor.fit(df[feature_cols])
num_names = num_cols
cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
all_feature_names = num_names + cat_names + bin_cols

X_scaled = preprocessor.transform(df[feature_cols])
print(f"Transformed feature matrix shape: {X_scaled.shape}")
print(f"Total features after encoding: {len(all_feature_names)}")

# PCA for 2D visualization & comparison
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(X_scaled)
var_explained = pca_2d.explained_variance_ratio_
print(f"2D PCA Explained Variance: PC1 = {var_explained[0]*100:.2f}%, PC2 = {var_explained[1]*100:.2f}% (Total = {sum(var_explained)*100:.2f}%)")

# -----------------------------------------------------------------------------
# 3. K-MEANS CLUSTERING & OPTIMAL K SELECTION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("3. K-MEANS CLUSTERING (ELBOW METHOD & SILHOUETTE ANALYSIS)")
print("=" * 80)

k_range = range(2, 11)
wcss = []
silhouette_scores_km = []
db_scores_km = []
ch_scores_km = []

for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=15, random_state=42)
    labels = km.fit_predict(X_scaled)
    wcss.append(km.inertia_)
    sil = silhouette_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    silhouette_scores_km.append(sil)
    db_scores_km.append(db)
    ch_scores_km.append(ch)
    print(f"k={k:2d} | WCSS (Inertia): {km.inertia_:10.2f} | Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | Calinski-Harabasz: {ch:8.2f}")

# Plot Elbow & Silhouette
fig, ax1 = plt.subplots(1, 2, figsize=(15, 5))

# Elbow plot (WCSS)
ax1[0].plot(k_range, wcss, marker='o', color='#1f77b4', linewidth=2, markersize=7)
ax1[0].set_title("K-Means Elbow Method (Inertia / WCSS)", fontsize=13, fontweight='bold')
ax1[0].set_xlabel("Number of Clusters (k)")
ax1[0].set_ylabel("Within-Cluster Sum of Squares (Inertia)")
ax1[0].grid(True, linestyle='--', alpha=0.6)
ax1[0].axvline(x=3, color='#d62728', linestyle='--', label='Elbow Point (k=3)')
ax1[0].legend(frameon=True)

# Silhouette Score plot
ax1[1].plot(k_range, silhouette_scores_km, marker='s', color='#2ca02c', linewidth=2, markersize=7)
ax1[1].set_title("K-Means Silhouette Score vs k", fontsize=13, fontweight='bold')
ax1[1].set_xlabel("Number of Clusters (k)")
ax1[1].set_ylabel("Average Silhouette Coefficient")
ax1[1].grid(True, linestyle='--', alpha=0.6)
best_k_sil = list(k_range)[np.argmax(silhouette_scores_km)]
ax1[1].axvline(x=best_k_sil, color='#d62728', linestyle='--', label=f'Max Silhouette (k={best_k_sil})')
ax1[1].legend(frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "kmeans_elbow_silhouette.png"), bbox_inches='tight')
plt.close()

# We select k=3 as optimal balance of segment interpretability, elbow inflection, and silhouette score
optimal_k = 3
print(f"\nOptimal K selected: {optimal_k}")

# Full K-Means Pipeline
kmeans_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, random_state=42))
])
kmeans_pipeline.fit(df[feature_cols])
kmeans_labels = kmeans_pipeline.named_steps['cluster'].labels_

df['cluster_kmeans'] = kmeans_labels

# -----------------------------------------------------------------------------
# 4. HIERARCHICAL (AGGLOMERATIVE) CLUSTERING & DENDROGRAM
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("4. HIERARCHICAL CLUSTERING (DENDROGRAM & LINKAGE COMPARISON)")
print("=" * 80)

# Compare Linkages
linkage_methods = ['ward', 'complete', 'average']
for method in linkage_methods:
    agg = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage=method)
    lbls = agg.fit_predict(X_scaled)
    s = silhouette_score(X_scaled, lbls)
    db = davies_bouldin_score(X_scaled, lbls)
    ch = calinski_harabasz_score(X_scaled, lbls)
    print(f"Linkage: {method:8s} (k={optimal_k}) -> Silhouette: {s:.4f} | Davies-Bouldin: {db:.4f} | Calinski-Harabasz: {ch:8.2f}")

# Generate Dendrogram with Ward Linkage (using sample of 120 customers for visual clarity)
np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled), size=120, replace=False)
X_sample = X_scaled[sample_indices]

Z = linkage(X_sample, method='ward', metric='euclidean')

plt.figure(figsize=(14, 6))
plt.title("Hierarchical Clustering Dendrogram (Ward's Minimum Variance)", fontsize=14, fontweight='bold')
plt.xlabel("Customer Sample Index", fontsize=11)
plt.ylabel("Ward Distance (Euclidean)", fontsize=11)
dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=8,
    color_threshold=18,
    above_threshold_color='#7f7f7f'
)
plt.axhline(y=18, color='#d62728', linestyle='--', linewidth=1.5, label='Cluster Cut Threshold (k=3)')
plt.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hierarchical_dendrogram.png"), bbox_inches='tight')
plt.close()

# Full Agglomerative Pipeline
hierarchical_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward'))
])
hierarchical_pipeline.fit(df[feature_cols])
hierarchical_labels = hierarchical_pipeline.named_steps['cluster'].labels_

df['cluster_hierarchical'] = hierarchical_labels

# -----------------------------------------------------------------------------
# 5. DBSCAN (DENSITY-BASED SPATIAL CLUSTERING OF APPLICATIONS WITH NOISE)
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("5. DBSCAN (K-DISTANCE GRAPH & DENSITY ESTIMATION)")
print("=" * 80)

# K-distance graph to determine optimal eps
# Rule of thumb for min_samples: min_samples >= 2 * dimensions (or min_samples >= 5)
k_neighbors = 5
nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
distances = np.sort(distances[:, k_neighbors - 1])

plt.figure(figsize=(9, 5))
plt.plot(distances, color='#9467bd', linewidth=2)
plt.title(f"DBSCAN K-Distance Graph (k = {k_neighbors}) for Optimal Epsilon", fontsize=13, fontweight='bold')
plt.xlabel("Data Points sorted by distance", fontsize=11)
plt.ylabel(f"{k_neighbors}-th Nearest Neighbor Distance", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=3.0, color='#d62728', linestyle='--', label=r'Selected $\varepsilon = 3.0$ (Elbow Region)')
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dbscan_kdistance.png"), bbox_inches='tight')
plt.close()

# Evaluate DBSCAN with eps=2.8, min_samples=4 (optimal density parameters for 27D space)
selected_eps = 2.8
selected_min_samples = 4

dbscan_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('cluster', DBSCAN(eps=selected_eps, min_samples=selected_min_samples))
])
dbscan_pipeline.fit(df[feature_cols])
dbscan_labels = dbscan_pipeline.named_steps['cluster'].labels_

df['cluster_dbscan'] = dbscan_labels

n_db_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_db_noise = (dbscan_labels == -1).sum()
pct_db_noise = (n_db_noise / len(df)) * 100

print(f"DBSCAN Parameters (Full Dim): eps = {selected_eps}, min_samples = {selected_min_samples}")
print(f"Number of clusters found: {n_db_clusters}")
print(f"Number of noise points (outliers): {n_db_noise} ({pct_db_noise:.2f}%)")

# Also run DBSCAN on PCA-reduced space (commonly used in industry for density clustering)
dbscan_pca = DBSCAN(eps=0.55, min_samples=10).fit(X_pca_2d)
df['cluster_dbscan_pca'] = dbscan_pca.labels_
n_db_pca_clusters = len(set(dbscan_pca.labels_)) - (1 if -1 in dbscan_pca.labels_ else 0)
n_db_pca_noise = (dbscan_pca.labels_ == -1).sum()
print(f"DBSCAN (PCA-2D projection) clusters: {n_db_pca_clusters}, noise: {n_db_pca_noise} ({n_db_pca_noise/len(df)*100:.2f}%)")

# -----------------------------------------------------------------------------
# 6. COMPREHENSIVE PERFORMANCE COMPARISON
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("6. PERFORMANCE EVALUATION COMPARISON")
print("=" * 80)

def compute_metrics(X, labels, name):
    mask = labels != -1  # Exclude noise for DBSCAN calculation
    n_c = len(set(labels[mask]))
    noise_count = (labels == -1).sum()
    if n_c > 1:
        sil = silhouette_score(X[mask], labels[mask])
        db = davies_bouldin_score(X[mask], labels[mask])
        ch = calinski_harabasz_score(X[mask], labels[mask])
    else:
        sil, db, ch = np.nan, np.nan, np.nan
    return {
        'Algorithm': name,
        'Clusters': n_c,
        'Noise Points': f"{noise_count} ({noise_count/len(labels)*100:.1f}%)" if noise_count > 0 else "0 (0.0%)",
        'Silhouette (High)': round(sil, 4) if not np.isnan(sil) else "N/A",
        'Davies-Bouldin (Low)': round(db, 4) if not np.isnan(db) else "N/A",
        'Calinski-Harabasz (High)': round(ch, 2) if not np.isnan(ch) else "N/A"
    }

results_table = pd.DataFrame([
    compute_metrics(X_scaled, kmeans_labels, "K-Means (k=3)"),
    compute_metrics(X_scaled, hierarchical_labels, "Hierarchical (Ward, k=3)"),
    compute_metrics(X_scaled, dbscan_labels, f"DBSCAN (eps={selected_eps}, ms={selected_min_samples})"),
    compute_metrics(X_pca_2d, dbscan_pca.labels_, "DBSCAN (2D PCA, eps=0.55)")
])

print(results_table.to_string(index=False))

# -----------------------------------------------------------------------------
# 7. VISUALIZATIONS: 2D PCA CLUSTER PROJECTIONS
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("7. GENERATING CLUSTERING COMPARISON VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# 1. K-Means
palette_km = sns.color_palette("Set2", n_colors=optimal_k)
sns.scatterplot(
    x=X_pca_2d[:, 0], y=X_pca_2d[:, 1],
    hue=kmeans_labels, palette=palette_km,
    ax=axes[0], s=35, alpha=0.85, edgecolor='none'
)
axes[0].set_title(f"K-Means (k={optimal_k})", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Principal Component 1")
axes[0].set_ylabel("Principal Component 2")
axes[0].legend(title="Cluster", frameon=True)

# 2. Hierarchical (Agglomerative)
sns.scatterplot(
    x=X_pca_2d[:, 0], y=X_pca_2d[:, 1],
    hue=hierarchical_labels, palette=palette_km,
    ax=axes[1], s=35, alpha=0.85, edgecolor='none'
)
axes[1].set_title(f"Hierarchical Agglomerative (k={optimal_k})", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Principal Component 1")
axes[1].legend(title="Cluster", frameon=True)

# 3. DBSCAN
unique_db_labels = sorted(list(set(dbscan_pca.labels_)))
db_palette = {lbl: sns.color_palette("tab10")[i % 10] if lbl != -1 else (0.7, 0.7, 0.7) for i, lbl in enumerate(unique_db_labels)}
sns.scatterplot(
    x=X_pca_2d[:, 0], y=X_pca_2d[:, 1],
    hue=dbscan_pca.labels_, palette=db_palette,
    ax=axes[2], s=35, alpha=0.85, edgecolor='none'
)
axes[2].set_title(f"DBSCAN (PCA Projection)", fontsize=13, fontweight='bold')
axes[2].set_xlabel("Principal Component 1")
axes[2].legend(title="Cluster / Noise", frameon=True)

plt.suptitle("Clustering Comparison: 2D Principal Component Projection", fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clusters_comparison_pca.png"), bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 8. CLUSTER PROFILING & BUSINESS INTERPRETATION
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("8. CUSTOMER CLUSTER PROFILING (K-MEANS)")
print("=" * 80)

key_profile_cols = [
    'annual_income', 'age', 'current_session_duration_sec', 'pages_viewed',
    'cart_items_count_this_session', 'cart_total_value', 'lifetime_purchases_count',
    'customer_conversion_rate', 'days_since_last_discount', 'is_returning_user',
    'sign_in', 'saw_checkout', 'should_receive_discount'
]

cluster_profile = df.groupby('cluster_kmeans')[key_profile_cols].mean().T
cluster_profile.columns = [f"Cluster {c}" for c in cluster_profile.columns]
cluster_sizes = df['cluster_kmeans'].value_counts().sort_index()
cluster_pcts = (cluster_sizes / len(df)) * 100

for c in range(optimal_k):
    cluster_profile.loc['Customer Count', f"Cluster {c}"] = int(cluster_sizes[c])
    cluster_profile.loc['Segment Share (%)', f"Cluster {c}"] = round(cluster_pcts[c], 1)

print(cluster_profile.round(2).to_string())

# Save Cluster Profile Heatmap
profile_normalized = df.groupby('cluster_kmeans')[key_profile_cols].mean()
profile_normalized = (profile_normalized - profile_normalized.min()) / (profile_normalized.max() - profile_normalized.min() + 1e-9)

plt.figure(figsize=(12, 6))
sns.heatmap(profile_normalized.T, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title("Normalized Feature Heatmap by K-Means Customer Segment", fontsize=14, fontweight='bold')
plt.xlabel("Customer Segment")
plt.ylabel("Behavioral & Demographic Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "cluster_profiles_heatmap.png"), bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# 9. EXPORTING FINAL ENRICHED DATASET
# -----------------------------------------------------------------------------
EXPORT_CSV = "customer_segmentation_with_clusters.csv"
df.to_csv(EXPORT_CSV, index=False)
print(f"\nFinal dataset with cluster labels saved to: {EXPORT_CSV}")
print("Workflow execution completed successfully!")
