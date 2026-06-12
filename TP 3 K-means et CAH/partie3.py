import pandas as pd

# Charger le dataset
df = pd.read_excel("studentData.xls")

# Afficher les premières lignes
# print(df.head())

# Agréger par étudiant
df_grouped = df.groupby("Name").agg({
    "Mark": "mean",
    "Attended": "mean"
}).reset_index()

# Renommer les colonnes
df_grouped.columns = ["Name", "Avg_Mark", "Avg_Attendance"]

# print(df_grouped)



from sklearn.preprocessing import StandardScaler

# Sélection des variables
X = df_grouped[["Avg_Mark", "Avg_Attendance"]]

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# print(X_scaled)


from sklearn.cluster import KMeans

# Choisir K=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Ajouter au dataset
df_grouped["Cluster"] = clusters

# print(df_grouped)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
sns.scatterplot(
    x=df_grouped["Avg_Attendance"],
    y=df_grouped["Avg_Mark"],
    hue=df_grouped["Cluster"],
    palette="viridis"
)

plt.title("Clustering K-Means")
plt.xlabel("Attendance")
plt.ylabel("Marks")

# plt.show()


from scipy.cluster.hierarchy import dendrogram, linkage

# Méthode complète
linked = linkage(X_scaled, method='complete')

plt.figure()
dendrogram(linked,
           labels=df_grouped["Name"].values,
           orientation='top')

plt.title("Dendrogramme")
# plt.show()


best_student = df_grouped.loc[df_grouped["Avg_Mark"].idxmax()]
# print("Meilleur étudiant :", best_student)

most_attended = df_grouped.loc[df_grouped["Avg_Attendance"].idxmax()]
# print("Plus assidu :", most_attended)

worst_student = df_grouped.loc[
    (df_grouped["Avg_Mark"] + df_grouped["Avg_Attendance"]).idxmin()
]
# print("Plus faible :", worst_student)

correlation = df_grouped["Avg_Mark"].corr(df_grouped["Avg_Attendance"])
# print("Corrélation :", correlation)

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_grouped[f"Cluster_{k}"] = kmeans.fit_predict(X_scaled)

# print(df_grouped)

from sklearn.metrics import silhouette_score

for k in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    # print(f"K={k}, Silhouette Score={score}")

import numpy as np

distances = kmeans.transform(X_scaled)
df_grouped["Min_Distance"] = np.min(distances, axis=1)

# Les plus proches des frontières
boundary_students = df_grouped.sort_values("Min_Distance").head()
# print(boundary_students)

plt.figure()
plt.scatter(
    df_grouped["Avg_Attendance"],
    df_grouped["Avg_Mark"],
    s=df_grouped["Avg_Mark"] * 10,
    c=df_grouped["Cluster"],
    cmap="viridis"
)
# plt.show()

centroids = kmeans.cluster_centers_

plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='red',
    marker='X',
    s=200
)

plt.title("Centroïdes")
# plt.show()

tutors = df_grouped[df_grouped["Cluster"] == 0]
print(tutors)