import numpy as np
import pandas as pd


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
    _weights : 1d-array
        Weights after fitting.
    _errors : list
        Number of miss classifications in each epoch.
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self._weights = None
        self._errors = None

    def fit(self, training_vectors, target_vectors):
        self._weights = np.zeros(1 + training_vectors.shape[1])
        self._errors = []

        for i in range(self.n_iter):
            errors = 0
            for training, target in zip(training_vectors, target_vectors):
                update = self.eta * (target - self.predict(training))
                self._weights[1:] += update * training
                self._weights[0] += update
                errors += int(update != 0.0)
            self._errors.append(errors)
        return self

    def net_input(self, x):
        return np.dot(x, self._weights[1:]) + self._weights[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
print(df.tail())
