# ============================================================
# EXERCICE 1 — Questions supplémentaires (version simple)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # changer en 'Agg' si erreur d'affichage
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import gower

# ── Préparation des données (commun à toutes les questions) ──
DF_cars = pd.read_excel("cars.xls")
DF_quant = DF_cars.select_dtypes(include=np.number)
scaler = StandardScaler()
cr_cars = scaler.fit_transform(DF_quant)
kmeans_ref = KMeans(n_clusters=3, n_init=5, random_state=42).fit(cr_cars)
labels_ref = kmeans_ref.labels_

# ============================================================
# Q1 — Gower + MDS puis K-means (alternative à PAM)
# ============================================================
print("=== Q1 — Gower + MDS vs K-means standard ===")

from sklearn.manifold import MDS

dis_cars = gower.gower_matrix(DF_cars.astype(object))
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords_gower = mds.fit_transform(dis_cars)
km_gower = KMeans(n_clusters=3, n_init=10, random_state=42).fit(coords_gower)

print("Clusters K-means standard :", dict(pd.Series(labels_ref).value_counts().sort_index()))
print("Clusters K-means Gower+MDS:", dict(pd.Series(km_gower.labels_).value_counts().sort_index()))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(DF_cars["Drive_Ratio"], DF_cars["Weight"], c=labels_ref, cmap='Set1', s=60)
ax1.set_title("K-means standard (vars quant. standardisées)")
ax1.set_xlabel("Drive_Ratio"); ax1.set_ylabel("Weight")

ax2.scatter(coords_gower[:, 0], coords_gower[:, 1], c=km_gower.labels_, cmap='Set1', s=60)
ax2.set_title("K-means sur Gower+MDS (données mixtes)")
ax2.set_xlabel("Dim MDS 1"); ax2.set_ylabel("Dim MDS 2")

plt.suptitle("Q1 — Comparaison K-means standard vs Gower")
plt.tight_layout()
plt.show()

# ============================================================
# Q2 — Standardisées vs non standardisées
# ============================================================
print("\n=== Q2 — Standardisées vs Non standardisées ===")

km_raw = KMeans(n_clusters=3, n_init=5, random_state=42).fit(DF_quant)

print("Clusters standardisés    :", dict(pd.Series(labels_ref).value_counts().sort_index()))
print("Clusters non standardisés:", dict(pd.Series(km_raw.labels_).value_counts().sort_index()))
print("Inertie standardisée    :", round(kmeans_ref.inertia_, 2))
print("Inertie non standardisée:", round(km_raw.inertia_, 2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(DF_cars["Drive_Ratio"], DF_cars["Weight"], c=labels_ref, cmap='Set1', s=60)
ax1.set_title("Données standardisées")
ax1.set_xlabel("Drive_Ratio"); ax1.set_ylabel("Weight")

ax2.scatter(DF_cars["Drive_Ratio"], DF_cars["Weight"], c=km_raw.labels_, cmap='Set1', s=60)
ax2.set_title("Données NON standardisées")
ax2.set_xlabel("Drive_Ratio"); ax2.set_ylabel("Weight")

plt.suptitle("Q2 — Impact de la standardisation")
plt.tight_layout()
plt.show()

# ============================================================
# Q3 — ACP + projection des clusters
# ============================================================
print("\n=== Q3 — ACP ===")

pca = PCA(n_components=2)
coords_pca = pca.fit_transform(cr_cars)

print("Variance expliquée : PC1 =", round(pca.explained_variance_ratio_[0]*100, 1), "%")
print("Variance expliquée : PC2 =", round(pca.explained_variance_ratio_[1]*100, 1), "%")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(coords_pca[:, 0], coords_pca[:, 1], c=labels_ref, cmap='Set1', s=80)
for i, nom in enumerate(DF_cars["Car"]):
    plt.annotate(nom[:8], (coords_pca[i, 0], coords_pca[i, 1]), fontsize=6, alpha=0.7)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Q3 — Projection ACP des clusters K-means")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()

# ============================================================
# Q4 — Effet de nstart
# ============================================================
print("\n=== Q4 — Effet de nstart ===")

import time

for n in [1, 5, 20, 50]:
    inerties = []
    t0 = time.time()
    for _ in range(10):
        km = KMeans(n_clusters=3, n_init=n).fit(cr_cars)
        inerties.append(km.inertia_)
    duree = time.time() - t0
    print(f"nstart={n:2d} | inertie moy={np.mean(inerties):.3f} | std={np.std(inerties):.3f} | temps={duree:.3f}s")

# ============================================================
# Q5 — Indices de validation
# ============================================================
print("\n=== Q5 — Indices de validation ===")

print(f"{'k':>3} | {'Silhouette':>12} | {'Calinski-H':>12} | {'Davies-B':>12}")
print("-" * 48)

for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(cr_cars)
    s  = silhouette_score(cr_cars, km.labels_)
    ch = calinski_harabasz_score(cr_cars, km.labels_)
    db = davies_bouldin_score(cr_cars, km.labels_)
    print(f"{k:>3} | {s:>12.4f} | {ch:>12.2f} | {db:>12.4f}")

# ============================================================
# Q6 — Distances intra et inter-cluster
# ============================================================
print("\n=== Q6 — Distances intra et inter-cluster ===")

centroids = kmeans_ref.cluster_centers_

print("Distances intra-cluster (moy distance au centroïde) :")
for cl in range(3):
    pts = cr_cars[labels_ref == cl]
    d_intra = np.mean(cdist(pts, [centroids[cl]]))
    print(f"  Cluster {cl} : {d_intra:.4f}")

print("\nDistances inter-cluster (entre centroïdes) :")
for i in range(3):
    for j in range(i+1, 3):
        d = np.linalg.norm(centroids[i] - centroids[j])
        print(f"  Cluster {i} <-> Cluster {j} : {d:.4f}")

# ============================================================
# Q7 — Statistiques descriptives par cluster
# ============================================================
print("\n=== Q7 — Profils par cluster ===")

DF_cars["Cluster"] = labels_ref

for cl in sorted(DF_cars["Cluster"].unique()):
    grp = DF_cars[DF_cars["Cluster"] == cl]
    print(f"\n--- Cluster {cl} ({len(grp)} véhicules) ---")
    print("Véhicules :", list(grp["Car"]))
    print(grp[list(DF_quant.columns)].describe().round(1).loc[["mean", "std", "min", "max"]])
