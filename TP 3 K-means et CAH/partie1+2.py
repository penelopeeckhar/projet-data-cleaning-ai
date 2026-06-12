import pandas as pd
data = pd.read_excel("CustomerDataSet.xls")
data.shape

# Afficher les premières lignes
print(data.head())

# Afficher les types des colonnes
print(data.dtypes)

# Statistiques descriptives
print(data.describe())

print("Attributs utilisables pour le clustering : ItemsBought, ItemsReturned")
print(data[['ItemsBought', 'ItemsReturned']])

# importer KMeans
from sklearn.cluster import KMeans

# importer matplotlib
import matplotlib.pyplot as plt

# importer preprocessing (StandardScaler pour la normalisation)
from sklearn import preprocessing

# créer le StandardScaler
scaler = preprocessing.StandardScaler()

# Copiez le dataframe avant le prétraitement afin que nous
# puissions accéder aux valeurs d'origine ultérieurement.
df2 = data.copy()

# préparer les deux attributs par le StandardScaler
df2[['ItemsBought', 'ItemsReturned']] = scaler.fit_transform(
    df2[['ItemsBought', 'ItemsReturned']]
)

# setup a figure
plt.figure(figsize=(14, 10))
ks = [2, 3, 4, 5, 6, 7]

# itérer sur la liste des valeurs de k
for idx, i in enumerate(ks, start=1):
    # créer le clusterer
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)

    # créer le regroupement
    labels = kmeans.fit_predict(df2[['ItemsBought', 'ItemsReturned']])

    # ajouter au subplot
    plt.subplot(3, 2, idx)
    plt.tight_layout(pad=3.0)

    # les étiquettes
    plt.title("#clusters (K) = {}".format(i))
    plt.xlabel('ItemsBought')
    plt.ylabel('ItemsReturned')

    # créer le scatter
    plt.scatter(df2['ItemsBought'], df2['ItemsReturned'],
                c=labels, cmap='tab10', s=80, edgecolors='k', linewidths=0.5)

# Afficher la figure
plt.show()

df2.head()


print("Valeur K retenue : K = 3")
print("Cluster 0 : Bons clients fidèles (beaucoup d'achats, peu de retours)")
print("Cluster 1 : Clients réguliers (achats et retours moyens)")
print("Cluster 2 : Clients insatisfaits (peu d'achats, beaucoup de retours)")

# Créer KMeans avec K = 3
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)

# Ajouter la colonne 'Cluster' au DataFrame original
data['Cluster'] = kmeans3.fit_predict(df2[['ItemsBought', 'ItemsReturned']])

# Ajouter aussi à df2 (normalisé) pour les graphiques suivants
df2['Cluster'] = data['Cluster']

# Afficher le résultat
print(data[['Customer ID', 'ItemsBought', 'ItemsReturned', 'Cluster']])

# Créer KMeans avec K = 3
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)

# Créer le clustering
labels = kmeans3.fit_predict(df2[['ItemsBought', 'ItemsReturned']])

# Créer un scatter
plt.figure(figsize=(9, 7))
plt.scatter(df2['ItemsBought'], df2['ItemsReturned'],
            c=labels, cmap='tab10', s=100, edgecolors='k', linewidths=0.6)

# annoter les produits en utilisant la colonne 'Product' et la fonction annotate
for i, row in data.iterrows():
    plt.annotate(str(row['Product']),
                 (df2.loc[i, 'ItemsBought'], df2.loc[i, 'ItemsReturned']),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)

# les étiquettes des axes
plt.xlabel('ItemsBought')
plt.ylabel('ItemsReturned')
plt.title('K-Means K=3 – Annotation par Product ID')

# Afficher
plt.show()

# Créer le scatter (données originales non normalisées)
plt.figure(figsize=(9, 7))
plt.scatter(data['ItemsBought'], data['ItemsReturned'],
            c=data['Cluster'], cmap='tab10', s=100, edgecolors='k', linewidths=0.6)

# Annoter chaque point de données en utilisant le code ZIP
for i, row in data.iterrows():
    plt.annotate(str(row['ZipCode']),
                 (row['ItemsBought'], row['ItemsReturned']),
                 textcoords="offset points", xytext=(5, 5), fontsize=8, color='darkblue')

# les étiquettes
plt.xlabel('ItemsBought')
plt.ylabel('ItemsReturned')
plt.title('K-Means K=3 – Annotation par ZipCode')

# Afficher le plot
plt.show()

# importer linkage et dendrogram from scipy
from scipy.cluster.hierarchy import linkage, dendrogram

# Créer le clustering
Z = linkage(df2[['ItemsBought', 'ItemsReturned']], method='ward')

# Afficher le dendrogram
plt.figure(figsize=(14, 6))
dendrogram(Z, labels=data['Customer ID'].tolist())

# Les étiquettes
plt.xlabel('Customer IDs')
plt.ylabel('distance')
plt.title('Dendrogramme – Classification Hiérarchique (Ward)')

# # Afficher
# plt.show()

from sklearn.cluster import AgglomerativeClustering

# setup a figure
plt.figure(figsize=(14, 10))

# itérer sur les différents nombres de clusters (ici: 3 et 4)
counter = 1
for i in [3, 4]:

    # ajouter un subplot
    plt.subplot(2, 2, counter)
    counter += 1

    # Les étiquettes
    plt.tight_layout(pad=3.0)
    plt.title('Dendrogram - {} clusters'.format(i))
    plt.xlabel('Count of Customers')
    plt.ylabel('distance')

    # Créer le clustering et afficher le dendrogram (tronqué)
    dendrogram(Z, truncate_mode='lastp', p=i, show_contracted=True)

    # Ajouter un second subplot (scatter)
    plt.subplot(2, 2, counter)
    counter += 1

    # Créer le clustering par troncature du dendrogram
    agg = AgglomerativeClustering(n_clusters=i)
    agg_labels = agg.fit_predict(df2[['ItemsBought', 'ItemsReturned']])

    plt.tight_layout(pad=3.0)
    plt.title('Regroupement hiérarchique – {} clusters'.format(i))
    plt.xlabel('ItemsBought')
    plt.ylabel('ItemsReturned')
    plt.scatter(data['ItemsBought'], data['ItemsReturned'],
                c=agg_labels, cmap='tab10', s=100, edgecolors='k', linewidths=0.6)

# Afficher la figure
plt.show()

# Charger le fichier StudentData.xls dans un DataFrame
sdata = pd.read_excel("StudentData.xls")

# Afficher les premiers enregistrements
sdata.head()
# Grouper le dataframe par étudiant (le nom de l'étudiant) et calculer les valeurs moyennes
student_agg = sdata.groupby('Name')[['Mark', 'Attended']].mean().reset_index()
student_agg.columns = ['Name', 'AvgMark', 'AvgAttended']

# Afficher les premiers enregistrements
student_agg.head(10)
# print(student_agg)

# Envisager un prétraitement (normalisation)
scaler2 = preprocessing.StandardScaler()
X_students = scaler2.fit_transform(student_agg[['AvgAttended', 'AvgMark']])

# Créer une instance de KMeans
kmeans_s = KMeans(n_clusters=3, random_state=42, n_init=10)

# Créer le clustering
student_agg['Cluster'] = kmeans_s.fit_predict(X_students)

# Afficher le scatter
plt.figure(figsize=(10, 8))
plt.scatter(student_agg['AvgAttended'], student_agg['AvgMark'],
            c=student_agg['Cluster'], cmap='Set1',
            s=120, edgecolors='k', linewidths=0.6)

# Les étiquettes
plt.xlabel("Attended classes")
plt.ylabel("Mark")
plt.title("K-Means K=3 – Profils d'étudiants")

# Annoter les données par les noms des étudiants
for _, row in student_agg.iterrows():
    plt.annotate(row['Name'],
                 (row['AvgAttended'], row['AvgMark']),
                 textcoords="offset points", xytext=(5, 4), fontsize=8)

# Afficher la figure
plt.show()

# lister les métriques à utiliser
modes = ['single', 'average', 'complete']

# créer une figure
plt.figure(figsize=(20, 5))
y_axis = None

# itérer sur la liste des métriques
for i, mode in enumerate(modes):

    # Créer un subplot
    y_axis = plt.subplot(1, 3, i + 1)

    # les étiquettes du subplot
    plt.title('Dendrogram - linkage mode: {}'.format(mode))
    plt.xlabel('ID of student')
    plt.ylabel('distance')

    # Créer le clustering
    Z_s = linkage(X_students, method=mode)

    # Afficher le dendrogram
    dendrogram(Z_s,
               labels=student_agg['Name'].apply(lambda x: x.split()[0]).tolist(),
               leaf_rotation=90,
               leaf_font_size=8)

# Afficher la figure
plt.tight_layout()
plt.show()

