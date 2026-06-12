import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from scipy.spatial.distance import pdist

import gower

# ============================================================
# CAH sur les données "fromage.txt"
# ============================================================

# 1. Charger les données
fromage = pd.read_csv("fromage.txt", sep="\t", decimal=".", index_col=0)

# 2. Résumé statistique
print(fromage.describe())

# 3. Graphe de corrélation
sns.pairplot(fromage)
plt.show()

# 4. Centrage-réduction
scaler = StandardScaler()
fromage_cr = scaler.fit_transform(fromage)

# 5. Matrice de distances
fromage_dist = pdist(fromage_cr, metric='euclidean')

# 6. Dendrogramme (méthode de Ward)
cah_ward = linkage(fromage_dist, method='ward')

# 7. Affichage
plt.figure(figsize=(10, 5))
dendrogram(cah_ward, labels=fromage.index)
plt.title("Dendrogramme - Ward")
plt.show()

# 8. Analyse des sauts d’inertie

# a) Les hauteurs (inerties)
inertie = np.sort(cah_ward[:, 2])[::-1]

plt.plot(inertie, drawstyle='steps-post')
plt.xlabel("Nombre de classes")
plt.ylabel("Inertie")
plt.show()

# b) Zoom sur les 15 premiers sauts
plt.plot(inertie[:15], drawstyle='steps-post')
plt.xlabel("Nombre de classes")
plt.ylabel("Inertie")
plt.show()

# c) Points de rupture
plt.plot(inertie[:15], drawstyle='steps-post')
plt.scatter([2,4,5], inertie[[2,4,5]], color=['green','red','blue'], s=80)
plt.show()

# d) Visualisation des partitions
# (équivalent de rect.hclust)

groupes_2 = cut_tree(cah_ward, n_clusters=2).flatten()
groupes_4 = cut_tree(cah_ward, n_clusters=4).flatten()
groupes_5 = cut_tree(cah_ward, n_clusters=5).flatten()

# e) Découpage final en 4 classes
groupes_cah = cut_tree(cah_ward, n_clusters=4).flatten()

# 9. Lister les groupes
res = pd.DataFrame({"Fromage": fromage.index, "Groupe": groupes_cah})
print(res.sort_values("Groupe"))