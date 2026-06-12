# ============================================================
# Imports nécessaires
# ============================================================

# Les librairies usuelles :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist

import gower   # pip install gower


# ============================================================
# EXERCICE 1 — K-means sur les données "cars.xls"
# ============================================================

# 1. Charger les données
DF_cars = pd.read_excel("cars.xls")

# 2. Résumé statistique
print(DF_cars.describe(include='all'))

# 3. Matrice de dispersion (pairs plot)
sns.pairplot(DF_cars)
plt.show()

# 4. Distance de Gower
DF_cars_gower = DF_cars.astype(object)
dis_cars = gower.gower_matrix(DF_cars_gower)

# Extraction des colonnes quantitatives
DF_quant = DF_cars.select_dtypes(include=np.number)

print("Colonnes quantitatives :", DF_quant.columns)

# centrage-réduction
scaler = StandardScaler()
cr_cars = scaler.fit_transform(DF_quant)

# 5. Segmentation k-means (3 classes)
kmeans_3 = KMeans(n_clusters=3, n_init=5).fit(cr_cars)

# 6. Afficher les résultats
print("Inertie :", kmeans_3.inertia_)
print("Centres :", kmeans_3.cluster_centers_)

# 7. Afficher les clusters
print(kmeans_3.labels_)
print(pd.Series(kmeans_3.labels_).value_counts())

# 8. Choix du nombre de clusters — méthode du coude
inertie_expl = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=5).fit(cr_cars)
    totss = km.inertia_
    inertie_expl.append(totss)

plt.plot(range(1, 11), inertie_expl, marker='o')
plt.xlabel("Nombre de groupes")
plt.ylabel("Inertie intra-classe")
plt.title("Méthode du coude")
plt.show()

# Version comparable à R (% inertie expliquée)
inertie_expl2 = []
totss = inertie_expl[0]

for k in range(2, 11):
    inertie_expl2.append(1 - inertie_expl[k-1] / totss)

plt.plot(range(2, 11), inertie_expl2, marker='o')
plt.xlabel("Nombre de groupes")
plt.ylabel("Proportion d'inertie expliquée")
plt.title("Méthode du coude (inertie expliquée)")
plt.show()

# 9. Segmentation finale
nbrClasse = 3
kmeans_final = KMeans(n_clusters=nbrClasse, n_init=5).fit(cr_cars)

# 10. Visualisation (Drive_Ratio vs Weight)
plt.scatter(DF_cars["Drive_Ratio"], DF_cars["Weight"], c=kmeans_final.labels_)
plt.xlabel("Drive Ratio")
plt.ylabel("Weight")
plt.title("Clusters K-means")
plt.show()