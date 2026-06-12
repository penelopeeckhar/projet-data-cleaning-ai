# ============================================================
# TP 7 : RNA — Dataset Titanic
# ============================================================

# ── 2. Importation du Dataset ──────────────────────────────
import pandas as pd

train = pd.read_csv(r'CHEMIN\train.csv')
print(train.head())

# ── 3. Prétraitement ───────────────────────────────────────

# Suppression des colonnes non pertinentes
train = train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Gestion des valeurs manquantes
train['Age'] = train['Age'].fillna(train['Age'].median())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
# Encodage
train = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)

# ── 4. Split et Normalisation ──────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = train.drop("Survived", axis=1)
y = train["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ── 5. Construction du Réseau de Neurones ──────────────────
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── 6. Entraînement du Modèle ──────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# ── 7. Visualisation des Courbes ───────────────────────────
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.title("Courbe de Loss")
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title("Courbe de Accuracy")
plt.show()

# ── 8. Évaluation sur le Test Set ──────────────────────────
model.evaluate(X_test, y_test)

# ── 9. Prédictions pour un Passager Fictif ─────────────────
import numpy as np

nouveau_passager = np.array([[3, 22, 1, 0, 7.25, 0, 1, 0]])
nouveau_passager_scaled = scaler.transform(nouveau_passager)
prediction = model.predict(nouveau_passager_scaled)
print("Probabilité de survie :", prediction[0][0])


# ============================================================
# QUESTIONS POUR LE COMPTE-RENDU
# ============================================================

# ── Q4 : Comparaison des architectures ────────────────────

# 1 seule couche cachée
model_1c = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])
model_1c.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_1c.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
_, acc_1c = model_1c.evaluate(X_test, y_test, verbose=0)
print(f"1 couche cachée : accuracy = {acc_1c:.4f}")

# 2 couches cachées
model_2c = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_2c.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_2c.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
_, acc_2c = model_2c.evaluate(X_test, y_test, verbose=0)
print(f"2 couches cachées : accuracy = {acc_2c:.4f}")

# 3 couches cachées
model_3c = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_3c.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_3c.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
_, acc_3c = model_3c.evaluate(X_test, y_test, verbose=0)
print(f"3 couches cachées : accuracy = {acc_3c:.4f}")

# ── Q5 : Fonctions d'activation ───────────────────────────
for activation in ['relu', 'tanh', 'elu']:
    m = Sequential([
        Dense(32, activation=activation, input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(16, activation=activation),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
    _, acc = m.evaluate(X_test, y_test, verbose=0)
    print(f"Activation {activation} : accuracy = {acc:.4f}")

# ── Q6 : Comparaison des optimizers ───────────────────────
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

for opt_name, opt in [('SGD', SGD()), ('Adam', Adam()), ('RMSprop', RMSprop())]:
    m = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
    _, acc = m.evaluate(X_test, y_test, verbose=0)
    print(f"Optimizer {opt_name} : accuracy = {acc:.4f}")

# ── Q7 : Taux d'apprentissage ─────────────────────────────
for lr in [0.1, 0.01, 0.001]:
    m = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    h = m.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
    _, acc = m.evaluate(X_test, y_test, verbose=0)
    print(f"lr={lr} : test_acc={acc:.4f}, val_acc_finale={h.history['val_accuracy'][-1]:.4f}")

# ── Q8 : Dropout et régularisation L2 ────────────────────
from tensorflow.keras.regularizers import l2

# Avec Dropout renforcé
model_dropout = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_dropout.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
_, acc_d = model_dropout.evaluate(X_test, y_test, verbose=0)
print(f"Dropout renforcé : accuracy = {acc_d:.4f}")

# Avec régularisation L2
model_l2 = Sequential([
    Dense(32, activation='relu', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(1, activation='sigmoid')
])
model_l2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_l2.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
_, acc_l2 = model_l2.evaluate(X_test, y_test, verbose=0)
print(f"Régularisation L2 : accuracy = {acc_l2:.4f}")

# ── Q9 : Batch Normalization ──────────────────────────────
from tensorflow.keras.layers import BatchNormalization

model_bn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dense(16, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])
model_bn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_bn.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0, validation_split=0.2)
_, acc_bn = model_bn.evaluate(X_test, y_test, verbose=0)
print(f"Batch Normalization : accuracy = {acc_bn:.4f}")

# ── Q10 : Matrice de confusion, F1, ROC, AUC ─────────────
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, classification_report

y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :\n", cm)

# F1 Score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score : {f1:.4f}")

# AUC
auc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC : {auc:.4f}")

# Courbe ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC")
plt.legend()
plt.show()

print(classification_report(y_test, y_pred))

# ── Q12 : Comparaison avec d'autres modèles ───────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest : {rf.score(X_test, y_test):.4f}")

# SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
print(f"SVM : {svm.score(X_test, y_test):.4f}")

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"KNN : {knn.score(X_test, y_test):.4f}")

_, rna_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"RNA : {rna_acc:.4f}")

# ── Q13 : SHAP ────────────────────────────────────────────
import shap

explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 50))
shap_values = explainer.shap_values(X_test[:50])

shap.summary_plot(shap_values, X_test[:50],
                  feature_names=X.columns.tolist())