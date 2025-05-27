import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
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

def print_raw_data(pandas_data):
    print(pandas_data.head())

def plot_raw_training_data(pandas_data):
    # grab first 100 records, and grab 0 and 2nd indexed values
    x = pandas_data.iloc[0:100, [0,2]].values
    # first 50 are setosa, next 50 are versicolor in the training data set
    pyplot.scatter(x[:50, 0], x[:50, 1], color='red', marker='o', label='setosa')
    pyplot.scatter(x[50:100, 0], x[50:100, 1], color='blue', marker='x', label='versicolor')
    pyplot.xlabel('sepal length')
    pyplot.ylabel('petal length')
    pyplot.legend(loc='upper left')
    pyplot.show()


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
print_raw_data(df)
plot_raw_training_data(df)