import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from matplotlib.colors import ListedColormap


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
        return self._errors

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

def plot_decision_regions(training, target, classifier, resolution=0.02):
    # set up symbology
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(target))])

    # plot the decision surface
    x1_min, x1_max = training[:, 0].min() - 1, training[:, 0].max() + 1
    x2_min, x2_max = training[:, 1].min() - 1, training[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    pyplot.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
    pyplot.xlim(xx1.min(), xx1.max())
    pyplot.ylim(xx2.min(), xx2.max())

    # plot class samples
    for i, cl in enumerate( np.unique(target)):
        pyplot.scatter(x=training[target == cl, 0], y=training[target == cl, 1], alpha=0.8, c=cmap(i), marker=markers[i], label=cl)



df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
print_raw_data(df)
plot_raw_training_data(df)


# all training set values
training_names = df.iloc[0:100, 4].values
training_values = df.iloc[0:100, [0, 2]].values
# construct an array of classifiers, set to -1 where setosa, 1 if versicolor
training_classifiers = np.where(training_names == "Iris-setosa", -1, 1)


perceptron = Perceptron(eta=0.1, n_iter=10)
plot_errors = perceptron.fit(training_values, training_classifiers)
pyplot.plot(range(1, len(plot_errors) + 1), plot_errors, marker='o')
pyplot.xlabel('Epochs')
pyplot.ylabel('# miss-classifications')
pyplot.show()

plot_decision_regions(training_values, training_classifiers, classifier=perceptron)
pyplot.xlabel('sepal length [cm]')
pyplot.ylabel('petal length [cm')
pyplot.legend(loc='upper left')
pyplot.show()