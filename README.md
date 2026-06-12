# Projet Data Cleaning & AI – ENSA Fès (GDNC4)

Ce dépôt regroupe l'ensemble des travaux pratiques (TP) réalisés dans le cadre du module **Machine Learning / Préparation des données**, du nettoyage de données (data cleaning) jusqu'aux réseaux de neurones, en passant par la classification, le clustering, la segmentation, la régression et l'évaluation de modèles.

Chaque dossier correspond à un TP indépendant, contenant le script Python (ou notebook), les jeux de données utilisés, ainsi que les rapports/supports fournis ou rendus.

## 📁 Structure du dépôt

```
projet-data-cleaning-ai/
├── TP 1/                                  → Prétraitement des données (R & Python)
├── TP 2 KNN/                              → Classification KNN sur le dataset Iris
├── TP 3 K-means et CAH/                   → Clustering (K-Means & CAH)
├── TP 4 Méthodes de Segmentation partie 1/→ Évaluation et sélection de modèles
├── TP 4 Méthodes de Segmentation partie 2/→ Segmentation avancée (exercices)
├── TP 5 ML/                               → Régression linéaire
└── TP 6/                                  → Réseaux de neurones (RNA) – Titanic
```

---

## 🧹 TP 1 — Prétraitement des données

**Fichiers :** `tp1.py`, `namefile.xlsx`, `TP1_ Praitraitement des données Langage R.docx`, rapport PDF.

Script Python (`tp1.py`) réalisant les premières étapes classiques de nettoyage d'un jeu de données :
- Exploration initiale (`head()`, `describe()`, `info()`)
- Visualisation par matrice de dispersion (`seaborn.pairplot`)
- Détection et suppression des colonnes invariantes (variance = 0)
- Suppression des doublons
- Détection des colonnes "éparses" (> 40 % de valeurs manquantes) et suppression
- Imputation des valeurs manquantes restantes

Un volet complémentaire en **R** est fourni dans le document Word, couvrant la même problématique de prétraitement.

---

## 🔍 TP 2 — Classification KNN (dataset Iris)

**Fichiers :** `tp2-partie1.py`, `tp2-partie2.py`, `02_knn_demo.ipynb`, `iris.csv`, `tp_knn_complet`, rapport PDF.

- **Partie 1** : implémentation "manuelle" du pipeline — extraction des caractéristiques (longueur/largeur des pétales), encodage des classes (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`), mélange aléatoire et split train/test (100/50).
- **Partie 2** : pipeline complet avec scikit-learn — `train_test_split`, normalisation via `StandardScaler`, entraînement d'un `KNeighborsClassifier` (k=5), puis évaluation (accuracy, matrice de confusion, rapport de classification).
- Le notebook `02_knn_demo.ipynb` propose une version interactive/démo du même algorithme.

---

## 📊 TP 3 — Clustering : K-Means et CAH

**Fichiers :** `partie1+2.py`, `partie3.py`, `CustomerDataSet.xls`, `StudentData.xls`, `TP_Clustering.ipynb`, rapports PDF.

- **Partie 1 & 2** : application de **K-Means** sur `CustomerDataSet.xls` (attributs `ItemsBought` / `ItemsReturned`), avec normalisation (`StandardScaler`) et recherche du nombre optimal de clusters (méthode du coude, k = 2 à 7).
- **Partie 3** : sur `StudentData.xls`, agrégation des données par étudiant (moyenne des notes et de l'assiduité), standardisation, puis clustering K-Means avec K=3 pour identifier des profils d'étudiants.
- Le notebook `TP_Clustering.ipynb` illustre également la **Classification Ascendante Hiérarchique (CAH)**.

---

## 🧪 TP 4 (Partie 1) — Évaluation et sélection de modèles

**Fichier principal :** `tp4_evaluation.py` (script structuré et commenté, réutilisant les datasets Iris/CustomerDataSet/StudentData des TP précédents).

Le script est organisé en 4 grandes parties :
1. **Métriques de classification (KNN / Iris)** : holdout simple, holdout stratifié, holdout répété (moyenne de l'accuracy sur k itérations), matrice de confusion, précision/rappel/F1-score, spécificité, validation croisée K-Fold, Leave-One-Out (LOO), sélection du meilleur `k` pour KNN via validation croisée.
2. **Métriques de régression** : MSE, RMSE, MAE, R² et R² ajusté.
3. **Évaluation sensible aux coûts** (analyse type détection de fraude sur `CustomerDataSet`).
4. **Comparaison KNN vs Clustering** via validation croisée.

---

## 🧩 TP 4 (Partie 2) — Méthodes de segmentation (exercices complémentaires)

**Fichiers :** `exercice1.py`, `exercice2.py`, `ex1_questions_simples.py`, `ex2_questions_simples.py`, `cars.xls`, `fromage.txt`, rapports PDF.

Série d'exercices appliquant les techniques de segmentation/clustering sur de nouveaux jeux de données (`cars.xls`, `fromage.txt`), avec des versions "questions simples" répondant point par point aux consignes du support de cours.

---

## 📈 TP 5 — Régression linéaire

**Fichier :** `exercice.py` + support et rapport PDF.

- **Partie 1** : régression linéaire simple (relation taille/poids) — nuage de points, modélisation avec `LinearRegression`, calcul du coefficient, de l'ordonnée à l'origine et du score R².
- Utilisation de `statsmodels` pour le calcul du **VIF (Variance Inflation Factor)**, en lien avec la détection de multicolinéarité, et de la validation croisée pour évaluer la performance du modèle (MSE).

---

## 🧠 TP 6 — Réseaux de neurones (RNA) — Dataset Titanic

**Fichier :** `ex.py` + dossier `titanic/` (données), support et rapport PDF.

Pipeline complet de Machine Learning appliqué à la prédiction de survie sur le Titanic :
1. **Importation** du dataset (`train.csv`)
2. **Prétraitement** : suppression des colonnes non pertinentes (`PassengerId`, `Name`, `Ticket`, `Cabin`), gestion des valeurs manquantes (médiane pour `Age`, mode pour `Embarked`), encodage one-hot des variables catégorielles (`Sex`, `Embarked`)
3. **Split & normalisation** des données (`train_test_split` + `StandardScaler`)
4. **Construction d'un réseau de neurones** avec **TensorFlow/Keras** (`Sequential`, couches `Dense` et `Dropout`) pour la classification binaire (survie / non-survie)

---

## 🛠️ Technologies utilisées

- **Python** : pandas, numpy, matplotlib, seaborn
- **Machine Learning** : scikit-learn (KNN, K-Means, métriques, preprocessing), statsmodels
- **Deep Learning** : TensorFlow / Keras
- **R** (TP1, volet prétraitement)
- **Jupyter Notebook**

## ▶️ Exécution

Chaque script peut être lancé indépendamment depuis le dossier du TP correspondant :

```bash
cd "TP 2 KNN"
python tp2-partie2.py
```

> ⚠️ Pour le TP6, le chemin du fichier `train.csv` est actuellement codé en dur (`C:\Users\saidm\...`) — à adapter selon votre environnement avant exécution.

## 📄 Rapports

Chaque dossier contient un rapport final au format PDF (`RAPPORT FINAL TPx.pdf`) détaillant la démarche, les résultats et l'interprétation des analyses pour le TP correspondant.

---

*Projet réalisé par Abir Majdi — 4ᵉ année Ingénierie du Développement Numérique et Cybersécurité, ENSA Fès.*
