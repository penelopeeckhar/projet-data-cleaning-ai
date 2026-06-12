# ============================================================
# EXERCICE 2 — Questions supplémentaires (version simple)
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
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree, cophenet
from scipy.spatial.distance import pdist

# ── Préparation des données (commun à toutes les questions) ──
fromage = pd.read_csv("fromage.txt", sep="\t", decimal=".", index_col=0)
scaler = StandardScaler()
fromage_cr = scaler.fit_transform(fromage)
fromage_dist = pdist(fromage_cr, metric='euclidean')
cah_ward = linkage(fromage_dist, method='ward')
groupes_ref = cut_tree(cah_ward, n_clusters=4).flatten()

# ============================================================
# Q1 — Comparaison des méthodes de linkage
# ============================================================
print("=== Q1 — Comparaison des méthodes de linkage ===")

methodes = ['ward', 'single', 'complete', 'average']

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, m in enumerate(methodes):
    cah = linkage(fromage_dist, method=m)
    dendrogram(cah, labels=fromage.index, ax=axes[i], leaf_rotation=45, leaf_font_size=8)
    axes[i].set_title(f"Méthode : {m.upper()}")

plt.suptitle("Q1 — Dendrogrammes selon la méthode de linkage")
plt.tight_layout()
plt.show()

# Taille des groupes à k=4
print("\nTaille des groupes k=4 selon la méthode :")
for m in methodes:
    cah = linkage(fromage_dist, method=m)
    grp = cut_tree(cah, n_clusters=4).flatten()
    print(f"  {m:10s} :", dict(pd.Series(grp).value_counts().sort_index()))

# ============================================================
# Q2 — Corrélation cophenétique
# ============================================================
print("\n=== Q2 — Corrélation cophenétique ===")

for m in methodes:
    cah = linkage(fromage_dist, method=m)
    c, _ = cophenet(cah, fromage_dist)
    print(f"  {m:10s} : {c:.4f}")

# ============================================================
# Q3 — ACP + projection des clusters CAH
# ============================================================
print("\n=== Q3 — ACP ===")

pca = PCA(n_components=2)
coords_pca = pca.fit_transform(fromage_cr)

print("Variance expliquée : PC1 =", round(pca.explained_variance_ratio_[0]*100, 1), "%")
print("Variance expliquée : PC2 =", round(pca.explained_variance_ratio_[1]*100, 1), "%")

colors = ['red', 'blue', 'green', 'orange']

plt.figure(figsize=(9, 7))
for g in range(4):
    mask = groupes_ref == g
    plt.scatter(coords_pca[mask, 0], coords_pca[mask, 1],
                c=colors[g], label=f'Groupe {g+1}', s=80)
    for idx in np.where(mask)[0]:
        plt.annotate(fromage.index[idx],
                     (coords_pca[idx, 0], coords_pca[idx, 1]),
                     fontsize=7, xytext=(4, 2), textcoords='offset points')

plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Q3 — Projection ACP des groupes CAH (k=4)")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# Q4 — Profils moyens pour k=2, 3, 4, 5
# ============================================================
print("\n=== Q4 — Profils moyens par nombre de classes ===")

for k in [2, 3, 4, 5]:
    grp = cut_tree(cah_ward, n_clusters=k).flatten()
    df_k = fromage.copy()
    df_k['Groupe'] = grp + 1
    print(f"\n--- k={k} groupes ---")
    print(df_k.groupby('Groupe').mean().round(1))

# Variable la plus discriminante
print("\nVariable la plus discriminante (k=4) :")
df4 = fromage.copy()
df4['Groupe'] = groupes_ref + 1
discrim = df4.groupby('Groupe').mean().std().sort_values(ascending=False)
print(discrim.round(2))

# ============================================================
# Q5 — Comparaison CAH vs K-means (k=4)
# ============================================================
print("\n=== Q5 — Comparaison CAH vs K-means ===")

km4 = KMeans(n_clusters=4, n_init=20, random_state=42).fit(fromage_cr)

print("Matrice de contingence CAH vs K-means :")
print(pd.crosstab(
    pd.Series(groupes_ref + 1, name='CAH'),
    pd.Series(km4.labels_ + 1, name='K-means')
))

ari = adjusted_rand_score(groupes_ref, km4.labels_)
print(f"\nAdjusted Rand Index : {ari:.4f}  (1 = accord parfait, 0 = aléatoire)")

# ============================================================
# Q6 — Silhouette pour estimer k optimal
# ============================================================
print("\n=== Q6 — Silhouette pour k optimal ===")

print(f"{'k':>3} | {'Silhouette CAH':>15} | {'Silhouette K-means':>18}")
print("-" * 42)

for k in range(2, 9):
    g_cah = cut_tree(cah_ward, n_clusters=k).flatten()
    g_km  = KMeans(n_clusters=k, n_init=10, random_state=42).fit(fromage_cr).labels_
    s_cah = silhouette_score(fromage_cr, g_cah)
    s_km  = silhouette_score(fromage_cr, g_km)
    print(f"{k:>3} | {s_cah:>15.4f} | {s_km:>18.4f}")

# ============================================================
# Q7 — Suppression du fromage atypique
# ============================================================
print("\n=== Q7 — Suppression du fromage atypique ===")

# Identifier le fromage le plus atypique (z-score max)
z = np.abs((fromage_cr))
outlier_idx = z.max(axis=1).argmax()
outlier_name = fromage.index[outlier_idx]
print(f"Fromage atypique : {outlier_name}")
print(f"Valeurs : {fromage.iloc[outlier_idx].to_dict()}")

# CAH sans ce fromage
fromage_sans = fromage.drop(index=outlier_name)
fromage_sans_cr = StandardScaler().fit_transform(fromage_sans)
fromage_sans_dist = pdist(fromage_sans_cr, metric='euclidean')
cah_sans = linkage(fromage_sans_dist, method='ward')
groupes_sans = cut_tree(cah_sans, n_clusters=4).flatten()

# Cophenétique avant / après
c_avant, _ = cophenet(cah_ward, fromage_dist)
c_apres, _ = cophenet(cah_sans, fromage_sans_dist)
print(f"\nCorrélation cophenétique avant : {c_avant:.4f}")
print(f"Corrélation cophenétique après : {c_apres:.4f}")

# Groupes avant / après
print("\nGroupes AVANT suppression (k=4) :")
for g in range(4):
    membres = list(fromage.index[groupes_ref == g])
    print(f"  Groupe {g+1} ({len(membres)}) :", membres)

print(f"\nGroupes APRÈS suppression de {outlier_name} (k=4) :")
for g in range(4):
    membres = list(fromage_sans.index[groupes_sans == g])
    print(f"  Groupe {g+1} ({len(membres)}) :", membres)

# Dendrogrammes avant / après
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
dendrogram(cah_ward, labels=fromage.index, ax=ax1, leaf_rotation=45, leaf_font_size=8)
ax1.set_title(f"Avant suppression (n=29)\ncoph={c_avant:.3f}")

dendrogram(cah_sans, labels=fromage_sans.index, ax=ax2, leaf_rotation=45, leaf_font_size=8)
ax2.set_title(f"Après suppression de {outlier_name} (n=28)\ncoph={c_apres:.3f}")

plt.suptitle("Q7 — Impact de la suppression du fromage atypique")
plt.tight_layout()
plt.show()
