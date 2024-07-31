
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification

# Datos linealmente separables
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# Crear el perceptrón y entrenar
perceptron = Perceptron()
perceptron.fit(X, y)

# Mostrar los datos y la línea de decisión
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xx = np.linspace(xmin, xmax, 100)
yy = -(perceptron.coef_[0][0] * xx + perceptron.intercept_) / perceptron.coef_[0][1]
plt.plot(xx, yy, 'k-')
plt.title("Datos linealmente separables")
plt.show()

# Datos no linealmente separables
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2)

# Crear el perceptrón y entrenar
perceptron = Perceptron()
perceptron.fit(X, y)

# Mostrar los datos y la línea de decisión
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xx = np.linspace(xmin, xmax, 100)
yy = -(perceptron.coef_[0][0] * xx + perceptron.intercept_) / perceptron.coef_[0][1]
plt.plot(xx, yy, 'k-')
plt.title("Datos no linealmente separables")
plt.show()
