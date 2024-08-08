import Perceptron as p
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection._split import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

df = pd.read_csv('.\data\LSTM.txtA', header=None)

df = shuffle(df)
print(df.head())

"""
5.1,3.8,1.9,0.4,Iris-setosa
4.8,3.0,1.4,0.3,Iris-setosa
5.1,3.8,1.6,0.2,Iris-setosa
4.6,3.2,1.4,0.2,Iris-setosa
5.3,3.7,1.5,0.2,Iris-setosa
5.0,3.3,1.4,0.2,Iris-setosa
7.0,3.2,4.7,1.4,Iris-versicolor
6.4,3.2,4.5,1.5,Iris-versicolor
6.9,3.1,4.9,1.5,Iris-versicolor
5.5,2.3,4.0,1.3,Iris-versicolor
6.5,2.8,4.6,1.5,Iris-versicolor
"""
# Splitting the data
X = df.iloc[:, 0:4].values # del elemento 0 al 4  [5.4 3.4 1.7 0.2] en un solo arreglo
y = df.iloc[:, 4].values  # del 4 en adelante ['Iris-setosa' 'Iris-virginica' 'Iris-virginica' 'Iris-setosa'.....] en un solo arreglo
print("x:")
print(X)
print("y:")
print(y)

print("[*] Segregando la data y los labels  de prueba")
train_data, test_data, train_labels, test_labels = train_test_split(
                            X, y, test_size=0.25)

print("train_data")
print(train_data)
print("test_data")
print( test_data)
print(" train_labels")
print(train_labels)
print("test_labels")
print(test_labels)

print("[*] Cambio los labels por 1 y -1  si es o no es Iris-setosa ")
train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

print('Train data:', train_data[0:2])
print('Train labels:', train_labels[0:5])

print('Test data:', test_data[0:2])
print('Test labels:', test_labels[0:5])

# fitting the perceptron
print("[*] Entrenando el Perceptron a un ritmo de 0.1 y con 10 iteraciones ")
perceptron = p.Perceptron(eta=0.1, n_iter=10)
perceptron.fit(train_data, train_labels)

#  Predicting the results

test_preds = perceptron.predict(test_data)

print('Test predicciones:',test_preds)

# Mesuring Performances
accuracy = accuracy_score(test_preds, test_labels)
print('Accuracy:', round(accuracy, 2) * 100, "%")