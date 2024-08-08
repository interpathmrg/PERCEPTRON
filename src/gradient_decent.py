# https://www.youtube.com/watch?v=jwStsp8JUPU

import numpy as np

# Activation Function
def sigmoid(w_sum):
    return 1/(1+np.exp(-w_sum))

# Get Prediction
def predict(features, weights, bias):
    return sigmoid(np.dot(features, weights) + bias)

# Loss Function
def cross_entropy(target, pred):
    return -(target*np.log10(pred)+(1-target)*(np.log10(1-pred)))

# Update Weights
def gradient_descent(x, y, weights, bias, learnrate, pred):
    new_weights = []

    bias += learnrate*( y - pred)

    for w,xi in zip(weights,x):
        new_weight = w + learnrate * (y - pred) * xi # formula de recalculo del nuevo peso
        new_weights.append(new_weight) 
    return new_weights, bias

# Data
features = np.array(([0.1,0.5,0.2],[0.2,0.3,0.1],[0.7,0.4,0.2],[0.1,0.4,0.3]))
targets = np.array([0,1,0,1])
epochs = 4000
learnrate = 0.1
# pesos iniciales
weights = np.array([0.4, 0.2, 0.6])
bias = 0.5 #Sesgo

errors = []
new_weights = []
for e in range(epochs):
    for x, y in zip(features, targets):
        pred = predict(x, weights, bias)
        error = cross_entropy(y, pred)
        weights, bias = gradient_descent(x, y, weights, bias, learnrate, pred)
    
    # Printing out the log-loss error on the training set
    out = predict(features, weights, bias)
    loss = np.mean(cross_entropy(targets, out))
    errors.append(loss)
    print("\n========== Epoch", e,"==========")
    print("New We: ", weights)
    print("Target: ", targets)
    print("Out   : ", out)
    print("Average loss: ", loss)