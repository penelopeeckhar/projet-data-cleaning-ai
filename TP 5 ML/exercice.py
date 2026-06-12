import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ============================================================
# PARTIE 1 : Régression linéaire simple
# ============================================================

data = {
    "taille": [150, 160, 170, 180, 190],
    "poids": [50, 60, 65, 75, 85]
}
df = pd.DataFrame(data)
X = df["taille"].values.reshape(-1, 1)
y = df["poids"].values

# Q1 - 5 premières lignes
print(df.head())

# Q2 - Nuage de points
plt.scatter(df["taille"], df["poids"], color='blue')
plt.xlabel("Taille (cm)")
plt.ylabel("Poids (kg)")
plt.title("Nuage de points : Taille vs Poids")
plt.show()

# ============================================================
# Régression avec scikit-learn
# ============================================================

model = LinearRegression()
model.fit(X, y)
print("Coefficient (pente) :", model.coef_[0])
print("Ordonnée à l'origine :", model.intercept_)
print("Score R² :", model.score(X, y))

# Q3 - Prédire pour taille = 175 cm
pred_175 = model.predict([[175]])
print("Poids prédit pour 175 cm :", pred_175[0])

# Visualisation droite de régression
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Taille (cm)")
plt.ylabel("Poids (kg)")
plt.title("Régression linéaire")
plt.show()

# ============================================================
# PARTIE 2 : Méthode analytique (formule matricielle)
# ============================================================

X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print("Theta (intercept, slope) :", theta)

# ============================================================
# PARTIE 3 : Analyse des résidus
# ============================================================

y_pred = model.predict(X)
residus = y - y_pred

plt.scatter(X, residus)
plt.axhline(0, color='red')
plt.xlabel("Taille")
plt.ylabel("Résidus")
plt.title("Analyse des résidus")
plt.show()

# Histogramme des résidus (Q5 supplémentaire)
plt.hist(residus, bins=5)
plt.title("Distribution des résidus")
plt.xlabel("Résidu")
plt.show()

# MSE (Q6 supplémentaire)
mse = mean_squared_error(y, y_pred)
print("MSE :", mse)

# ============================================================
# PARTIE 4 : Régression linéaire multiple
# ============================================================

data = {
    "taille": [150, 160, 170, 180, 190],
    "age":    [20,  25,  30,  35,  40],
    "poids":  [50,  60,  65,  75,  85]
}
df = pd.DataFrame(data)
X = df[["taille", "age"]]
y = df["poids"]

model = LinearRegression()
model.fit(X, y)
print("Coefficients :", model.coef_)
print("Intercept :", model.intercept_)
print("Score R² :", model.score(X, y))

# Q9 - Matrice de corrélation
print(df.corr())

# Q10 - Modèle sans "age"
X_sans_age = df[["taille"]]
model2 = LinearRegression()
model2.fit(X_sans_age, y)
print("R² sans age :", model2.score(X_sans_age, y))

# Q11 - VIF
X_vif = df[["taille", "age"]]
for i, col in enumerate(X_vif.columns):
    vif = variance_inflation_factor(X_vif.values, i)
    print(f"VIF {col} : {vif:.2f}")

# ============================================================
# PARTIE 5 : Validation croisée
# ============================================================

scores = cross_val_score(LinearRegression(), X, y, cv=3, scoring='neg_mean_squared_error')
print("Scores :", scores)
print("Score moyen :", scores.mean())

# Q7 - Comparaison avec modèle naïf (prédire la moyenne)
y_naif = np.full(len(y), y.mean())
mse_naif = mean_squared_error(y, y_naif)
print("MSE modèle naïf :", mse_naif)
print("MSE modèle linéaire :", mse)

# Q8 - Extrapolation (taille = 210 cm)
pred_210 = model2.predict([[210]])
print("Poids prédit pour 210 cm :", pred_210[0])