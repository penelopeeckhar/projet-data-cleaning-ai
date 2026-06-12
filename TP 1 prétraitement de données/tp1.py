import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.impute import IterativeImputer, KNNImputer

df = pd.read_excel("namefile.xlsx")
print("Afficher les premières lignes : ",df.head())

print("Statistiques générales : ",df.describe())
print("Infos générales : ",df.info())

# Matrice de dispersion
sns.pairplot(df)
plt.show()

print("Données invariantes (variance = 0)")
print("Calcul de la variance :")
df.var(numeric_only=True)
print("Supprimer les colonnes avec variance = 0")
df = df.loc[:, df.nunique() > 1]
print("Colonnes après suppression des invariantes :")
print(df.columns)

print("Supprimer les doublons")
df = df.drop_duplicates()
print(df.columns)

# Données éparses (> 40% NA)
print("Pourcentage de valeurs manquantes")
na_percent = df.isna().mean()
print(na_percent)
print("Colonnes avec plus de 40%")
sparse_columns = na_percent[na_percent > 0.4]
print(sparse_columns)
print("Supprimer si nécessaire")
df = df.loc[:, na_percent <= 0.4]
print(df.columns)

# Imputation des valeurs manquantes
# Méthode 1 : kNN
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns
imputer_knn = KNNImputer(n_neighbors=5)
df_numeric_imputed = pd.DataFrame(
    imputer_knn.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)
df_knn = pd.concat([df_numeric_imputed, df[categorical_cols]], axis=1)
print("Nombre de valeurs manquantes AVANT KNN :")
print(df.isna().sum())

print("Nombre de valeurs manquantes APRÈS KNN :")
print(df_knn.isna().sum())

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Séparer colonnes numériques
numeric_cols = df.select_dtypes(include=['number']).columns
# Initialiser l'imputer
imputer_mf = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=200),
    max_iter=10,
    random_state=42
)
# Appliquer sur données numériques
df_numeric_mf = pd.DataFrame(
    imputer_mf.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)
# Recombiner avec colonnes catégorielles
categorical_cols = df.select_dtypes(exclude=['number']).columns
df_mf = pd.concat([df_numeric_mf, df[categorical_cols]], axis=1)

print("AVANT MISSFOREST")
print(df.isna().sum())
print("APRES MISSFOREST")
print(df_mf.isna().sum())

df_locf = df.fillna(method='ffill')
df_locf = df_locf.fillna(method='bfill')
print(df_locf)


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer  # OBLIGATOIRE
from sklearn.impute import IterativeImputer

#Copier les données originales
df_original = df.copy()
# Créer artificiellement 10% de valeurs manquantes
df_test = df_original.copy()
# Séparer colonnes numériques et catégorielles
numeric_cols = df_test.select_dtypes(include=['number']).columns
categorical_cols = df_test.select_dtypes(exclude=['number']).columns
# Créer masque seulement pour colonnes numériques
mask = np.random.rand(*df_test[numeric_cols].shape) < 0.1
df_test.loc[:, numeric_cols] = df_test[numeric_cols].mask(mask)
print("Nombre de valeurs manquantes créées :")
print(df_test[numeric_cols].isna().sum())

# Imputation MICE (seulement numérique)
imputer_mice = IterativeImputer(random_state=42)
df_numeric_mice = pd.DataFrame(
    imputer_mice.fit_transform(df_test[numeric_cols]),
    columns=numeric_cols
)
# Recombiner avec colonnes catégorielles
df_mice_test = pd.concat([df_numeric_mice, df_test[categorical_cols]], axis=1)

# Imputation par la moyenne
df_numeric_mean = df_test[numeric_cols].fillna(df_test[numeric_cols].mean())
df_mean_test = pd.concat([df_numeric_mean, df_test[categorical_cols]], axis=1)

# Fonction erreur (MSE)
def erreur_imputation(original, imputed):
    mask = ~np.isnan(original)
    return mean_squared_error(original[mask], imputed[mask])

# Calcul des erreurs (seulement numérique)
erreur_mice = erreur_imputation(
    df_original[numeric_cols].values,
    df_mice_test[numeric_cols].values
)
erreur_mean = erreur_imputation(
    df_original[numeric_cols].values,
    df_mean_test[numeric_cols].values
)

# Affichage final
print("\n===== Comparaison des erreurs =====")
print("Erreur MICE :", erreur_mice)
print("Erreur Moyenne :", erreur_mean)

if erreur_mice < erreur_mean:
    print("MICE est plus précise que l'imputation par la moyenne.")
else:
    print("L'imputation par la moyenne est plus précise.")


# Détection des outliers pour la variable age
Q1_age = df["age"].quantile(0.25)
Q3_age = df["age"].quantile(0.75)
IQR_age = Q3_age - Q1_age

borne_inf_age = Q1_age - 1.5 * IQR_age
borne_sup_age = Q3_age + 1.5 * IQR_age

outliers_age = df[(df["age"] < borne_inf_age) | 
                  (df["age"] > borne_sup_age)]

print("===== Outliers détectés pour age =====")
print(outliers_age)

# Détection des outliers pour la variable tauxmax
Q1_taux = df["tauxmax"].quantile(0.25)
Q3_taux = df["tauxmax"].quantile(0.75)
IQR_taux = Q3_taux - Q1_taux

borne_inf_taux = Q1_taux - 1.5 * IQR_taux
borne_sup_taux = Q3_taux + 1.5 * IQR_taux

outliers_taux = df[(df["tauxmax"] < borne_inf_taux) | 
                   (df["tauxmax"] > borne_sup_taux)]

print("===== Outliers détectés pour tauxmax =====")
print(outliers_taux)

# Indices des outliers
indices_age = set(outliers_age.index)
indices_taux = set(outliers_taux.index)

# Intersection : outliers présents dans les deux variables
atyp_intersection = indices_age.intersection(indices_taux)

print("===== Outliers communs (age ET tauxmax) =====")
print(atyp_intersection)

import matplotlib.pyplot as plt

plt.scatter(df["age"], df["tauxmax"], label="Observations normales")

# Mettre en rouge les outliers communs
for idx in atyp_intersection:
    plt.scatter(df.loc[idx, "age"], 
                df.loc[idx, "tauxmax"], 
                color="red", 
                marker="*",
                s=150)

plt.xlabel("Age")
plt.ylabel("Tauxmax")
plt.title("Détection des outliers multivariés")
plt.legend()
plt.show()

print("Remplacement des outliers par kNN")

from sklearn.impute import KNNImputer

# Copier le dataset
df_clean = df.copy()

# Remplacer les outliers par NaN
df_clean.loc[list(atyp_intersection), "age"] = np.nan
df_clean.loc[list(atyp_intersection), "tauxmax"] = np.nan

# Séparer colonnes numériques
numeric_cols = df_clean.select_dtypes(include=['number']).columns

# Appliquer kNN
imputer_knn = KNNImputer(n_neighbors=5)
df_clean[numeric_cols] = imputer_knn.fit_transform(df_clean[numeric_cols])

print("===== Données après remplacement des outliers =====")
print(df_clean.loc[list(atyp_intersection)])