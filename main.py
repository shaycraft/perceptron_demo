import numpy as np


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    
    Attributes
    ----------
    weights : 1d-array
        Weights after fitting.
    errors : list
        Number of misclassifications in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.weights = np.zeros(1+X.shape[1])