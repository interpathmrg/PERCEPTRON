
import numpy as np


"""
η (eta): the learning rate is usually a small value between 0.0 and 1.0 which defines how quickly the model learns.
n_iter: the number of iterations.
w_: the weights. A weight defines the importance of the corresponding input value.
"""

class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter