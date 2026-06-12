"""
TP4 – Évaluation et Sélection du Modèle
=========================================
Basé sur : Partie 4 – Evaluation et sélection du modèle
Données   : Iris dataset (TP2 KNN) + CustomerDataSet / StudentData (TP3 Clustering)

Plan :
  PARTIE 1 – Métriques de classification (KNN / Iris)
    1.1  Holdout  (train/test split)
    1.2  Holdout stratifié  (stratify=)
    1.3  Holdout répété  (k itérations, moyenne ACC)
    1.4  Matrice de confusion + Accuracy
    1.5  Précision, Rappel, F1-score
    1.6  Spécificité
    1.7  Validation croisée K-Fold
    1.8  Leave-One-Out (LOO)
    1.9  Sélection du meilleur k (KNN) via CV

  PARTIE 2 – Métriques de régression (TP2/TP3 données)
    2.1  MSE, RMSE, MAE
    2.2  R² et R² ajusté

  PARTIE 3 – Évaluation sensible aux coûts (fraude / CustomerDataSet)

  PARTIE 4 – Comparaison KNN vs clustering (TP3) via CV
"""

# ─────────────────────────────────────────────────
# Imports communs
# ─────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, LeaveOneOut,
                                     RepeatedStratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report,
                             precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error,
                             r2_score)
from sklearn.linear_model import LinearRegression

# ─────────────────────────────────────────────────
# Chargement des données – iris.csv (TP2)
# ─────────────────────────────────────────────────
import os

# Cherche iris.csv dans le même dossier que ce script, ou dans les sous-dossiers courants
script_dir = os.path.dirname(os.path.abspath(__file__))
iris_candidates = [
    os.path.join(script_dir, 'iris.csv'),
    os.path.join(script_dir, 'TP 2 KNN', 'iris.csv'),
    'iris.csv',
]
iris_path = next((p for p in iris_candidates if os.path.exists(p)), None)
if iris_path:
    df_iris = pd.read_csv(iris_path)
    feature_cols  = ['SepalLength[cm]', 'SepalWidth[cm]', 'PetalLength[cm]', 'PetalWidth[cm]']
    X_iris        = df_iris[feature_cols].values
    y_raw         = df_iris['Species'].astype(str).values
    le            = LabelEncoder()
    y_iris        = le.fit_transform(y_raw)
    class_names   = le.classes_          # array de str propres
    feature_names = feature_cols
else:
    # Fallback sklearn si iris.csv introuvable
    from sklearn.datasets import load_iris as _load_iris
    _iris         = _load_iris()
    X_iris        = _iris.data
    y_iris        = _iris.target
    class_names   = np.array([str(c) for c in _iris.target_names])
    feature_names = _iris.feature_names
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)
print("=" * 65)
print("  TP4 – ÉVALUATION ET SÉLECTION DU MODÈLE")
print("=" * 65)
print(f"\nDataset Iris : {X_iris.shape[0]} exemples, {X_iris.shape[1]} features")
print(f"Classes      : {[str(c) for c in class_names]}\n")

# ═══════════════════════════════════════════════════════════════════
# PARTIE 1 – MÉTRIQUES DE CLASSIFICATION (KNN / IRIS)
# ═══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  PARTIE 1 – MÉTRIQUES DE CLASSIFICATION (KNN / IRIS)")
print("=" * 65)

# ───────────────────────────────────────────────────────────────────
# 1.1  HOLDOUT (split simple 70/30)
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.1  Holdout (70/30, sans stratification) ───")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_iris, test_size=0.30, random_state=42, shuffle=True
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc_holdout = accuracy_score(y_test, y_pred)
print(f"  Taille ensemble d'entraînement : {len(X_train)}")
print(f"  Taille ensemble de test        : {len(X_test)}")
print(f"  Accuracy (Holdout)             : {acc_holdout * 100:.2f}%")

# ───────────────────────────────────────────────────────────────────
# 1.2  HOLDOUT STRATIFIÉ
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.2  Holdout Stratifié (70/30, stratify=y) ───")

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
    X_scaled, y_iris, test_size=0.30, random_state=42, stratify=y_iris
)

knn_s = KNeighborsClassifier(n_neighbors=5)
knn_s.fit(X_tr_s, y_tr_s)
acc_strat = accuracy_score(y_te_s, knn_s.predict(X_te_s))

# Distribution des classes
unique, counts_all   = np.unique(y_iris,  return_counts=True)
unique, counts_train = np.unique(y_tr_s,  return_counts=True)
unique, counts_test  = np.unique(y_te_s,  return_counts=True)

print(f"  Distribution (total)  : {dict(zip([str(c) for c in class_names], counts_all.tolist()))}")
print(f"  Distribution (train)  : {dict(zip([str(c) for c in class_names], counts_train.tolist()))}")
print(f"  Distribution (test)   : {dict(zip([str(c) for c in class_names], counts_test.tolist()))}")
print(f"  Accuracy (stratifié)  : {acc_strat * 100:.2f}%")

# ───────────────────────────────────────────────────────────────────
# 1.3  HOLDOUT RÉPÉTÉ (k=10 répétitions)
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.3  Holdout Répété (k=10 itérations) ───")

k_rep = 10
acc_list = []

for i in range(k_rep):
    Xtr, Xte, ytr, yte = train_test_split(
        X_scaled, y_iris, test_size=0.30, random_state=i
    )
    m = KNeighborsClassifier(n_neighbors=5)
    m.fit(Xtr, ytr)
    acc_list.append(accuracy_score(yte, m.predict(Xte)))

acc_moyenne = (1 / k_rep) * sum(acc_list)   # formule du cours : ACC = (1/k) * Σ ACCi
print(f"  Accuracies par itération : {[f'{a*100:.1f}%' for a in acc_list]}")
print(f"  ACC moyenne (formule cours) = (1/{k_rep}) × Σ ACCi = {acc_moyenne * 100:.2f}%")

# ───────────────────────────────────────────────────────────────────
# 1.4  MATRICE DE CONFUSION + ACCURACY
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.4  Matrice de Confusion & Accuracy ───")

# Re-utilise le split stratifié
y_pred_s = knn_s.predict(X_te_s)
cm = confusion_matrix(y_te_s, y_pred_s)

# Calcul manuel de l'accuracy (formule du cours : ACC = (TP+TN) / All)
tp_tn = np.trace(cm)   # somme de la diagonale = toutes les prédictions correctes
total = cm.sum()
acc_manual = tp_tn / total

print(f"\n  Matrice de confusion :\n{cm}")
print(f"\n  Accuracy = (TP+TN) / All = {tp_tn} / {total} = {acc_manual * 100:.2f}%")

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Matrice de Confusion – KNN (k=5)\nHoldout Stratifié 70/30')
axes[0].set_xlabel('Prédictions')
axes[0].set_ylabel('Réel')

# ───────────────────────────────────────────────────────────────────
# 1.5  PRÉCISION, RAPPEL, F1-SCORE
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.5  Précision, Rappel, F1-Score ───")
print("\n  Formules du cours :")
print("    Précision = TP / (TP + FP)")
print("    Rappel    = TP / (TP + FN)")
print("    F1        = 2 × (Rappel × Précision) / (Rappel + Précision)")
print("              = 2TP / (2TP + FN + FP)")
prec   = precision_score(y_te_s, y_pred_s, average='macro')
rappel = recall_score   (y_te_s, y_pred_s, average='macro')
f1     = f1_score       (y_te_s, y_pred_s, average='macro')
print(f"\n  [Macro-average sur les 3 classes]")
print(f"  Précision : {prec * 100:.2f}%")
print(f"  Rappel    : {rappel * 100:.2f}%")
print(f"  F1-Score  : {f1 * 100:.2f}%")
# Calcul par classe (Iris est multi-classe → one-vs-rest)
print("\n  Détail par classe :")
for idx, cname in enumerate(class_names):
    # Pour chaque classe : binariser les labels
    y_bin_true = (y_te_s   == idx).astype(int)
    y_bin_pred = (y_pred_s == idx).astype(int)
    TP = ((y_bin_true == 1) & (y_bin_pred == 1)).sum()
    FP = ((y_bin_true == 0) & (y_bin_pred == 1)).sum()
    FN = ((y_bin_true == 1) & (y_bin_pred == 0)).sum()
    TN = ((y_bin_true == 0) & (y_bin_pred == 0)).sum()
    p  = TP / (TP + FP) if (TP + FP) > 0 else 0
    r  = TP / (TP + FN) if (TP + FN) > 0 else 0
    f  = (2 * r * p) / (r + p) if (r + p) > 0 else 0
    print(f"    {cname:<20} | TP={TP} FP={FP} FN={FN} TN={TN}"
          f" | Précision={p:.2f}  Rappel={r:.2f}  F1={f:.2f}")

# ───────────────────────────────────────────────────────────────────
# 1.6  SPÉCIFICITÉ
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.6  Spécificité ───")
print("  Formule du cours : Spécificité = TN / (TN + FP)")

print("\n  Spécificité par classe :")
for idx, cname in enumerate(class_names):
    y_bin_true = (y_te_s   == idx).astype(int)
    y_bin_pred = (y_pred_s == idx).astype(int)

    FP = ((y_bin_true == 0) & (y_bin_pred == 1)).sum()
    TN = ((y_bin_true == 0) & (y_bin_pred == 0)).sum()

    spec = TN / (TN + FP) if (TN + FP) > 0 else 0
    print(f"    {cname:<20} | TN={TN} FP={FP} | Spécificité = {TN} / ({TN}+{FP}) = {spec * 100:.2f}%")

# ───────────────────────────────────────────────────────────────────
# 1.7  VALIDATION CROISÉE K-FOLD (k=5 et k=10)
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.7  Validation Croisée K-Fold ───")
print("  Principe : données divisées en k parties égales,")
print("             chaque partie sert une fois comme ensemble de test.")

for k_fold in [5, 10]:
    cv = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    scores_cv = cross_val_score(
        KNeighborsClassifier(n_neighbors=5), X_scaled, y_iris,
        cv=cv, scoring='accuracy'
    )
    acc_cv = scores_cv.mean()
    print(f"\n  K-Fold (k={k_fold}) :")
    print(f"    Scores par fold : {[f'{s*100:.1f}%' for s in scores_cv]}")
    print(f"    ACC = (1/{k_fold}) × Σ ACCi = {acc_cv * 100:.2f}%  (±{scores_cv.std()*100:.2f}%)")

# ───────────────────────────────────────────────────────────────────
# 1.8  LEAVE-ONE-OUT (LOO)
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.8  Leave-One-Out (LOO) ───")
print("  Principe : 1 seul exemple utilisé pour la validation à chaque itération.")

loo = LeaveOneOut()
scores_loo = cross_val_score(
    KNeighborsClassifier(n_neighbors=5), X_scaled, y_iris,
    cv=loo, scoring='accuracy'
)
print(f"  Nombre d'itérations  : {len(scores_loo)}  (= nb d'exemples)")
print(f"  ACC moyenne (LOO)    : {scores_loo.mean() * 100:.2f}%")

# ───────────────────────────────────────────────────────────────────
# 1.9  SÉLECTION DU MEILLEUR K (KNN) PAR VALIDATION CROISÉE
# ───────────────────────────────────────────────────────────────────
print("\n─── 1.9  Sélection du meilleur k (KNN) via CV (k-fold=5) ───")
print("  (Sélection de modèle par variation d'hyperparamètre)")

k_range = range(1, 21)
cv_scores = []
cv_stds   = []

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k in k_range:
    s = cross_val_score(
        KNeighborsClassifier(n_neighbors=k), X_scaled, y_iris,
        cv=cv5, scoring='accuracy'
    )
    cv_scores.append(s.mean())
    cv_stds.append(s.std())

best_k   = k_range[np.argmax(cv_scores)]
best_acc = max(cv_scores)
print(f"  Meilleur k = {best_k}  →  ACC = {best_acc * 100:.2f}%")

# ═══════════════════════════════════════════════════════════════════
# PARTIE 2 – MÉTRIQUES DE RÉGRESSION
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PARTIE 2 – MÉTRIQUES DE RÉGRESSION")
print("=" * 65)
print("  Contexte : Prédire la largeur des pétales (PetalWidth)")
print("             à partir des autres features Iris\n")

# Régression : prédire PetalWidth (index 3) depuis les 3 autres features
X_reg = X_iris[:, [0, 1, 2]]   # SepalLength, SepalWidth, PetalLength
y_reg = X_iris[:, 3]           # PetalWidth

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.30, random_state=42
)

reg = LinearRegression()
reg.fit(X_reg_train, y_reg_train)
y_reg_pred = reg.predict(X_reg_test)

# ─── MSE et RMSE ───
mse  = mean_squared_error(y_reg_test, y_reg_pred)
rmse = np.sqrt(mse)

# ─── MAE ───
mae = mean_absolute_error(y_reg_test, y_reg_pred)

# ─── R² ───
r2 = r2_score(y_reg_test, y_reg_pred)

# ─── RSE (Relative Squared Error) = SCR / SCT ───
SCR = np.sum((y_reg_test - y_reg_pred) ** 2)
SCT = np.sum((y_reg_test - y_reg_test.mean()) ** 2)
RSE = SCR / SCT
r2_from_rse = 1 - RSE   # formule du cours : R² = 1 - RSE
# ─── R² ajusté ───
m = len(y_reg_test)          # nombre de points de données
n = X_reg_test.shape[1]      # nombre de variables
r2_adj = 1 - ((1 - r2) * (m - 1)) / (m - n - 1)

print("─── 2.1  MSE, RMSE, MAE ───")
print(f"  Formule MSE  = (1/n) × Σ (h(xi) - yi)²")
print(f"  MSE          = {mse:.6f}")
print(f"  RMSE         = √MSE = {rmse:.6f}")
print(f"\n  Formule MAE  = (1/n) × Σ |h(xi) - yi|")
print(f"  MAE          = {mae:.6f}")
print(f"\n  [MSE pénalise les grosses erreurs ; MAE est plus robuste aux outliers]")
print("\n─── 2.2  R² et R² Ajusté ───")
print(f"  RSE = SCR / SCT = {SCR:.6f} / {SCT:.6f} = {RSE:.6f}")
print(f"  R²  = 1 - RSE                         = {r2_from_rse:.6f}")
print(f"  R²  (sklearn)                          = {r2:.6f}")
print(f"\n  Formule R² ajusté = 1 - [(1-R²)(m-1)] / (m-n-1)")
print(f"    m={m} points, n={n} variables")
print(f"  R² ajusté = {r2_adj:.6f}")

if r2 >= 0.9:
    interp = "Très bon ajustement — le modèle explique quasi toute la variance."
elif r2 >= 0.7:
    interp = "Bon ajustement."
elif r2 >= 0.5:
    interp = "Ajustement modéré."
elif r2 >= 0:
    interp = "Faible ajustement — le modèle apporte peu d'information."
else:
    interp = "R² négatif : mauvais modèle, pire que la moyenne."
print(f"  Interprétation : {interp}")

# ═══════════════════════════════════════════════════════════════════
# PARTIE 3 – ÉVALUATION SENSIBLE AUX COÛTS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PARTIE 3 – ÉVALUATION SENSIBLE AUX COÛTS")
print("=" * 65)
print("  Contexte : Détection de fraude (CustomerDataSet.xls – TP3)")

# ── Chargement CustomerDataSet.xls (TP3) ──
customer_candidates = [
    os.path.join(script_dir, 'CustomerDataSet.xls'),
    os.path.join(script_dir, 'ABIR MAJDI TP3 GDNC', 'CustomerDataSet.xls'),
    'CustomerDataSet.xls',
]
customer_path = next((p for p in customer_candidates if os.path.exists(p)), None)
if customer_path:
    df_customer = pd.read_excel(customer_path)
    print(f"  CustomerDataSet chargé : {df_customer.shape[0]} clients, colonnes : {df_customer.columns.tolist()}")
else:
    print("  [CustomerDataSet.xls introuvable – matrices de confusion simulées comme dans le cours]")
    df_customer = None

# Matrice de coûts (du cours)
cost_matrix = {
    ('fraude', 'fraude')    : -1,   # TP (bonne détection = bénéfice)
    ('fraude', 'normale')   : 100,  # FN (fraude manquée = très coûteux)
    ('normale', 'fraude')   : 20,   # FP (fausse alarme = peu coûteux)
    ('normale', 'normale')  : 0     # TN
}

print("\n  Matrice de coûts :")
print(f"  {'':25} {'Prédit Fraude':>15} {'Prédit Normale':>15}")
print(f"  {'Réel  Fraude':25} {cost_matrix[('fraude','fraude')]:>15} {cost_matrix[('fraude','normale')]:>15}")
print(f"  {'Réel  Normale':25} {cost_matrix[('normale','fraude')]:>15} {cost_matrix[('normale','normale')]:>15}")

# Modèle 1 (du cours) : Accuracy=94%, Coût=5465
cm_m1 = np.array([[40, 55], [5, 900]])
acc_m1 = (cm_m1[0,0] + cm_m1[1,1]) / cm_m1.sum()
cost_m1 = (cm_m1[0,0] * cost_matrix[('fraude','fraude')]  +
           cm_m1[0,1] * cost_matrix[('fraude','normale')]  +
           cm_m1[1,0] * cost_matrix[('normale','fraude')]  +
           cm_m1[1,1] * cost_matrix[('normale','normale')])

# Modèle 2 (du cours) : Accuracy=64%, Coût=2310
cm_m2 = np.array([[40, 20], [340, 600]])
acc_m2 = (cm_m2[0,0] + cm_m2[1,1]) / cm_m2.sum()
cost_m2 = (cm_m2[0,0] * cost_matrix[('fraude','fraude')]  +
           cm_m2[0,1] * cost_matrix[('fraude','normale')]  +
           cm_m2[1,0] * cost_matrix[('normale','fraude')]  +
           cm_m2[1,1] * cost_matrix[('normale','normale')])

print(f"\n  ── Modèle 1 ──")
print(f"  Matrice : {cm_m1.tolist()}")
print(f"  Accuracy = {acc_m1 * 100:.0f}%")
print(f"  Coût     = 40×(-1) + 55×100 + 5×20 + 900×0 = {cost_m1}")

print(f"\n  ── Modèle 2 ──")
print(f"  Matrice : {cm_m2.tolist()}")
print(f"  Accuracy = {acc_m2 * 100:.0f}%")
print(f"  Coût     = 40×(-1) + 20×100 + 340×20 + 600×0 = {cost_m2}")

meilleur = 1 if cost_m1 < cost_m2 else 2
meilleur_cost = min(cost_m1, cost_m2)
meilleur_acc  = acc_m1 if meilleur == 1 else acc_m2
pire_acc      = acc_m2 if meilleur == 1 else acc_m1
pire_cost     = max(cost_m1, cost_m2)
print(f"\n  Conclusion : le Modèle {meilleur} (Accuracy={meilleur_acc*100:.0f}%, Coût={meilleur_cost})")
print(f"  est préférable au modèle adverse (Accuracy={pire_acc*100:.0f}%, Coût={pire_cost}).")
print(f"  → Un coût plus bas l'emporte, même si l'accuracy est plus faible.")
print("  → L'accuracy seule est trompeuse pour les problèmes à classes déséquilibrées.")

# ═══════════════════════════════════════════════════════════════════
# PARTIE 4 – COMPARAISON KNN vs CLUSTERING (TP2 / TP3)
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("  PARTIE 4 – COMPARAISON KNN : TP2 vs MEILLEUR K (CV)")
print("=" * 65)

print("\n  Méthode : Holdout répété (RepeatedStratifiedKFold, 5×5)")
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

results = {}
for k in [3, 5, 7, best_k]:
    s = cross_val_score(
        KNeighborsClassifier(n_neighbors=k), X_scaled, y_iris,
        cv=rkf, scoring='accuracy'
    )
    results[k] = s

print(f"\n  {'k (voisins)':<15} {'ACC moy':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("  " + "-" * 55)
for k, s in results.items():
    flag = " ← TP2 (k=5)" if k == 5 else (f" ← meilleur k (CV)" if k == best_k and k != 5 else "")
    print(f"  {k:<15} {s.mean()*100:>9.2f}% {s.std()*100:>9.2f}% "
          f"{s.min()*100:>9.2f}% {s.max()*100:>9.2f}%{flag}")

# ─── Rapport de classification final (meilleur k) ───
print(f"\n─── Rapport de classification final (k={best_k}, holdout stratifié 70/30) ───")
X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
    X_scaled, y_iris, test_size=0.30, random_state=42, stratify=y_iris
)
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_tr_f, y_tr_f)
y_pred_f = knn_best.predict(X_te_f)
print(classification_report(y_te_f, y_pred_f, target_names=class_names))

# ═══════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════

# ── Figure 1 : Distribution des classes (stratification) ──
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
fig1.suptitle('1.2 – Stratification : distribution des classes', fontsize=13, fontweight='bold')

for ax, counts, title in zip(
    axes1,
    [counts_all, counts_train, counts_test],
    ['Dataset complet (150)', 'Ensemble entraînement (105)', 'Ensemble test (45)']
):
    bars = ax.bar(class_names, counts / counts.sum() * 100, color=['#4C72B0','#DD8452','#55A868'])
    ax.set_title(title)
    ax.set_ylabel('Proportion (%)')
    ax.set_ylim(0, 50)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('fig1_stratification.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 2 : Matrice de confusion ──
fig2, ax2 = plt.subplots(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax2,
            linewidths=0.5, linecolor='gray')
ax2.set_title(f'1.4 – Matrice de Confusion\nKNN (k=5), Holdout Stratifié 70/30\nAccuracy = {acc_strat*100:.1f}%',
              fontsize=11, fontweight='bold')
ax2.set_xlabel('Prédictions')
ax2.set_ylabel('Classes réelles')
plt.tight_layout()
plt.savefig('fig2_matrice_confusion.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 3 : Précision / Rappel / F1 par classe ──
metrics_data = {}
for idx, cname in enumerate(class_names):
    y_bin_true = (y_te_s == idx).astype(int)
    y_bin_pred = (y_pred_s == idx).astype(int)
    TP = ((y_bin_true==1)&(y_bin_pred==1)).sum()
    FP = ((y_bin_true==0)&(y_bin_pred==1)).sum()
    FN = ((y_bin_true==1)&(y_bin_pred==0)).sum()
    p = TP/(TP+FP) if (TP+FP)>0 else 0
    r = TP/(TP+FN) if (TP+FN)>0 else 0
    f = (2*r*p)/(r+p) if (r+p)>0 else 0
    metrics_data[cname] = {'Précision': p, 'Rappel': r, 'F1-Score': f}

df_metrics = pd.DataFrame(metrics_data).T
fig3, ax3 = plt.subplots(figsize=(8, 5))
x = np.arange(len(class_names))
width = 0.25
colors = ['#4C72B0', '#DD8452', '#55A868']
for i, (col, color) in enumerate(zip(df_metrics.columns, colors)):
    ax3.bar(x + i * width, df_metrics[col], width, label=col, color=color, alpha=0.85)
ax3.set_xticks(x + width)
ax3.set_xticklabels(class_names)
ax3.set_ylim(0, 1.15)
ax3.set_ylabel('Score')
ax3.set_title('1.5 – Précision, Rappel et F1-Score par classe\n(KNN k=5, Holdout Stratifié)',
              fontweight='bold')
ax3.legend()
ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.4)
for i, (col, color) in enumerate(zip(df_metrics.columns, colors)):
    for j, val in enumerate(df_metrics[col]):
        ax3.text(j + i * width, val + 0.02, f'{val:.2f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('fig3_precision_rappel_f1.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 4 : Validation croisée K-Fold (k=5) ──
cv_fold_scores = {}
for k_fold in [2, 3, 5, 10, 20]:
    cv_tmp = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    s = cross_val_score(KNeighborsClassifier(n_neighbors=5),
                        X_scaled, y_iris, cv=cv_tmp, scoring='accuracy')
    cv_fold_scores[k_fold] = s

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))
fig4.suptitle('1.7 – Validation Croisée K-Fold', fontsize=13, fontweight='bold')

# Boxplot des scores
axes4[0].boxplot(
    [cv_fold_scores[k] for k in cv_fold_scores],
    labels=[f'k={k}' for k in cv_fold_scores],
    patch_artist=True,
    boxprops=dict(facecolor='#4C72B0', alpha=0.7)
)
axes4[0].set_title('Distribution des scores par k-fold')
axes4[0].set_ylabel('Accuracy')
axes4[0].set_xlabel('Nombre de folds')
axes4[0].axhline(y=acc_strat, color='red', linestyle='--', alpha=0.6, label=f'Holdout={acc_strat*100:.1f}%')
axes4[0].legend()

# Convergence des moyennes
means = [cv_fold_scores[k].mean() for k in cv_fold_scores]
stds  = [cv_fold_scores[k].std()  for k in cv_fold_scores]
k_list = list(cv_fold_scores.keys())
axes4[1].errorbar(k_list, [m*100 for m in means], [s*100 for s in stds],
                  marker='o', capsize=5, color='#DD8452', linewidth=2)
axes4[1].set_title('Accuracy moyenne selon le nombre de folds')
axes4[1].set_xlabel('Nombre de folds (k)')
axes4[1].set_ylabel('Accuracy (%)')
axes4[1].set_xticks(k_list)
axes4[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig4_kfold_cv.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 5 : Sélection du meilleur k (KNN) ──
fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.errorbar(list(k_range), [s * 100 for s in cv_scores],
             [s * 100 for s in cv_stds],
             marker='o', capsize=4, color='#4C72B0', linewidth=2)
ax5.axvline(x=best_k, color='red', linestyle='--', alpha=0.8)
ax5.annotate(f'Meilleur k={best_k}\n({best_acc*100:.1f}%)',
             xy=(best_k, best_acc * 100),
             xytext=(best_k + 1, best_acc * 100 - 1.5),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red')
ax5.set_xlabel('Valeur de k (voisins)')
ax5.set_ylabel('Accuracy CV (%)')
ax5.set_title('1.9 – Sélection du meilleur k par Validation Croisée (5-Fold)\n'
              'Sélection de modèle par variation d\'hyperparamètre', fontweight='bold')
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig5_best_k_selection.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 6 : Métriques de régression ──
fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
fig6.suptitle('Partie 2 – Métriques de Régression\n(Prédiction PetalWidth, Régression Linéaire)',
              fontsize=12, fontweight='bold')

# Scatter réel vs prédit
axes6[0].scatter(y_reg_test, y_reg_pred, alpha=0.7, color='#4C72B0')
axes6[0].plot([y_reg_test.min(), y_reg_test.max()],
              [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
axes6[0].set_xlabel('Valeurs réelles')
axes6[0].set_ylabel('Valeurs prédites')
axes6[0].set_title(f'Réel vs Prédit\nR²={r2:.4f}')

# Résidus
residuals = y_reg_test - y_reg_pred
axes6[1].scatter(y_reg_pred, residuals, alpha=0.7, color='#DD8452')
axes6[1].axhline(0, color='red', linestyle='--')
axes6[1].set_xlabel('Valeurs prédites')
axes6[1].set_ylabel('Résidus')
axes6[1].set_title('Résidus')

# Tableau récapitulatif des métriques
axes6[2].axis('off')
table_data = [
    ['Métrique', 'Valeur', 'Usage'],
    ['MSE',      f'{mse:.5f}',  'Comparaison modèles'],
    ['RMSE',     f'{rmse:.5f}', 'Comparaison modèles'],
    ['MAE',      f'{mae:.5f}',  'Robuste aux outliers'],
    ['R²',       f'{r2:.5f}',   'Ajustement modèle'],
    ['R² ajusté',f'{r2_adj:.5f}','Sélection variables'],
    ['RSE',      f'{RSE:.5f}',  'Erreur relative'],
]
table = axes6[2].table(cellText=table_data[1:], colLabels=table_data[0],
                       cellLoc='center', loc='center',
                       colColours=['#4C72B0','#4C72B0','#4C72B0'])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)
axes6[2].set_title('Récapitulatif métriques', fontweight='bold')

plt.tight_layout()
plt.savefig('fig6_regression_metrics.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 7 : Évaluation sensible aux coûts ──
fig7, axes7 = plt.subplots(1, 3, figsize=(15, 5))
fig7.suptitle('Partie 3 – Évaluation Sensible aux Coûts\n(Détection de Fraude)',
              fontsize=12, fontweight='bold')

labels_cm = ['Fraude', 'Normale']
for ax, cm_data, title, acc, cost in zip(
    axes7[:2],
    [cm_m1, cm_m2],
    ['Modèle 1\n(Accuracy haute, Coût élevé)', 'Modèle 2\n(Accuracy basse, Coût faible)'],
    [acc_m1, acc_m2],
    [cost_m1, cost_m2]
):
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels_cm, yticklabels=labels_cm, ax=ax,
                linewidths=0.5)
    ax.set_title(f'{title}\nAcc={acc*100:.0f}%  |  Coût={cost}', fontsize=10)
    ax.set_xlabel('Prédit')
    ax.set_ylabel('Réel')

# Comparaison Accuracy vs Coût
axes7[2].axis('off')
cmp_data = [
    ['Critère',    'Modèle 1', 'Modèle 2'],
    ['Accuracy',   f'{acc_m1*100:.0f}%',   f'{acc_m2*100:.0f}%'],
    ['Coût total', f'{cost_m1}',            f'{cost_m2}'],
    ['Décision',   '❌ Rejeté',             '✅ Préféré'],
]
tbl = axes7[2].table(cellText=cmp_data[1:], colLabels=cmp_data[0],
                     cellLoc='center', loc='center',
                     colColours=['#4C72B0','#DD8452','#55A868'])
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(1.5, 2.2)
axes7[2].set_title('Accuracy ≠ Meilleur modèle\n(Coût total décisif)', fontweight='bold')

plt.tight_layout()
plt.savefig('fig7_cost_sensitive.png', dpi=120, bbox_inches='tight')
plt.close()

# ── Figure 8 : Comparaison des méthodes d'évaluation ──
fig8, ax8 = plt.subplots(figsize=(10, 6))

methods = ['Holdout\n(70/30)', 'Holdout\nStratifié', 'Holdout\nRépété (×10)',
           'K-Fold\n(k=5)', 'K-Fold\n(k=10)', 'LOO', f'Meilleur\nk={best_k}']
accs = [acc_holdout, acc_strat, acc_moyenne,
        cross_val_score(KNeighborsClassifier(n_neighbors=5), X_scaled, y_iris,
                        cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean(),
        cross_val_score(KNeighborsClassifier(n_neighbors=5), X_scaled, y_iris,
                        cv=StratifiedKFold(10, shuffle=True, random_state=42)).mean(),
        scores_loo.mean(),
        cross_val_score(KNeighborsClassifier(n_neighbors=best_k), X_scaled, y_iris,
                        cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean()]

colors_bar = ['#4C72B0','#4C72B0','#4C72B0','#DD8452','#DD8452','#55A868','#C44E52']
bars = ax8.bar(methods, [a * 100 for a in accs], color=colors_bar, alpha=0.85,
               edgecolor='black', linewidth=0.5)
ax8.set_ylabel('Accuracy (%)')
ax8.set_title('Récapitulatif : Comparaison de toutes les méthodes d\'évaluation\n'
              '(KNN k=5 sauf dernière barre)', fontweight='bold')
ax8.set_ylim(92, 100)
ax8.axhline(y=95, color='red', linestyle='--', alpha=0.4, label='Seuil 95%')
ax8.legend()

for bar, v in zip(bars, accs):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{v*100:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

import matplotlib.patches as mpatches
legend_elements = [
    mpatches.Patch(color='#4C72B0', label='Méthodes Holdout'),
    mpatches.Patch(color='#DD8452', label='K-Fold CV'),
    mpatches.Patch(color='#55A868', label='LOO'),
    mpatches.Patch(color='#C44E52', label='Meilleur k (CV)'),
]
ax8.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig('fig8_methodes_comparaison.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n" + "=" * 65)
print("  RÉSUMÉ FINAL")
print("=" * 65)
print(f"\n  Méthode Holdout (70/30)            : {acc_holdout*100:.2f}%")
print(f"  Holdout Stratifié                  : {acc_strat*100:.2f}%")
print(f"  Holdout Répété (k=10)              : {acc_moyenne*100:.2f}%")
print(f"  K-Fold CV (k=5)                    : {cross_val_score(KNeighborsClassifier(n_neighbors=5),X_scaled,y_iris,cv=StratifiedKFold(5,shuffle=True,random_state=42)).mean()*100:.2f}%")
print(f"  K-Fold CV (k=10)                   : {cross_val_score(KNeighborsClassifier(n_neighbors=5),X_scaled,y_iris,cv=StratifiedKFold(10,shuffle=True,random_state=42)).mean()*100:.2f}%")
print(f"  Leave-One-Out                      : {scores_loo.mean()*100:.2f}%")
print(f"  Meilleur k trouvé par CV           : k = {best_k}  ({best_acc*100:.2f}%)")
print(f"\n  MSE (régression)   : {mse:.5f}")
print(f"  RMSE               : {rmse:.5f}")
print(f"  MAE                : {mae:.5f}")
print(f"  R²                 : {r2:.5f}")
print(f"  R² ajusté          : {r2_adj:.5f}")
print(f"\n  Coût Modèle 1 (Acc={acc_m1*100:.0f}%) : {cost_m1}  ← préféré (coût minimal)")
print(f"  Coût Modèle 2 (Acc={acc_m2*100:.0f}%) : {cost_m2}")
print("\n  Toutes les figures sauvegardées dans le dossier du script.")
