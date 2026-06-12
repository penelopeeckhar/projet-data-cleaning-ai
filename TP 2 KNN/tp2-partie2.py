import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target
# print("X",X)
# print("Y",y)
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# print("X_train",X_train)
# print("y_train",y_train)
# print("X_test",X_test)
# print("y_test",y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# print("X_train_scaled",X_train_scaled)
# print("X_test_scaled",X_test_scaled)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
# print("y_pred ",y_pred)

# print("Accuracy :", accuracy_score(y_test, y_pred))
# print("\nMatrice de confusion :\n", confusion_matrix(y_test, y_pred))
# print("\nRapport de classification :\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.xlabel("Prédictions")
plt.ylabel("Réel")
# plt.show()

scores = []

k_values = range(1,21)
for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    scores.append(
        accuracy_score(y_test, knn.predict(X_test_scaled))
    )

plt.plot(k_values, scores, marker='o')
plt.xlabel("Valeur de k")
plt.ylabel("Précision")
plt.title("Choix du meilleur k")
plt.show()
print("Meilleur k :", k_values[np.argmax(scores)])