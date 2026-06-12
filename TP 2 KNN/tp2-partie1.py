import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_iris = pd.read_csv('iris.csv')
# print(df_iris.tail())

X = df_iris[['PetalLength[cm]', 'PetalWidth[cm]']].values
X[:5, :]
# print("Extraction des caractéristiques",X)

label_dict = {'Iris-setosa': 0,
              'Iris-versicolor': 1,
              'Iris-virginica': 2}

df_iris['ClassLabel'] = df_iris['Species'].map(label_dict)
# print("Extraction des classes",df_iris.tail())

y = df_iris['ClassLabel'].values
# print(y[:5])

indices = np.arange(y.shape[0])
# print(indices)

rnd = np.random.RandomState(123)
shuffled_indices = rnd.permutation(indices)
# print(shuffled_indices)

X_shuffled, y_shuffled = X[shuffled_indices], y[shuffled_indices]
X_train, y_train = X_shuffled[:100], y_shuffled[:100]
X_test, y_test = X_shuffled[100:], y_shuffled[100:]
# print("X_train",X_train)
# print("y_train",y_train)
# print("X_test",X_test)
# print("y_test",y_test)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
X, y = iris.data[:, 2:], iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    random_state=123,
                                                    shuffle=True)
# print("X_train",X_train)
# print("y_train",y_train)
# print("X_test",X_test)
# print("y_test",y_test)

plt.scatter(X_train[y_train == 0, 0],
            X_train[y_train == 0, 1],
            marker='o',
            label='class 0 (Setosa)')

plt.scatter(X_train[y_train == 1, 0],
            X_train[y_train == 1, 1],
            marker='^',
            label='class 1 (Versicolor)')

plt.scatter(X_train[y_train == 2, 0],
            X_train[y_train == 2, 1],
            marker='s',
            label='class 2 (Virginica)')

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')

# plt.show()

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
print(knn_model.fit(X_train, y_train))

y_pred = knn_model.predict(X_test)
num_correct_predictions = (y_pred == y_test).sum()
accuracy = (num_correct_predictions / y_test.shape[0]) * 100
# print('Test set accuracy: %.2f%%' % accuracy)

from mlxtend.plotting import plot_decision_regions


plot_decision_regions(X_train, y_train, knn_model)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
# plt.show()

plot_decision_regions(X_test, y_test, knn_model)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()