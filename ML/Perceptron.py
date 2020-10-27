import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron:
    def __init__(self, scale: float, max_epochs: int):
        """
        Initializes the Perceptron object
        :param scale: The scale at which the weights of the classifier are manipulated.
        :param max_epochs: The maximum number of times the fit function will run if the classifier cannot converge.
        """
        self._scale = scale
        self._max_epochs = max_epochs
        self._weights = []

    def _predict(self, x):
        summation = self._net_input(x)
        summation = np.where(summation > 0, 1, -1)
        return summation

    def _net_input(self, x: np.ndarray):
        """
        Applies the weights to each datapoint for classification in the step activation function.
        :param x: The nd array of datapoints
        :return: The value of each datapoint after weights are applied.
        """
        return np.dot(x, self._weights[1:]) + self._weights[0]

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Trains the perceptron.
        :param x: The datapoints to train on.
        :param y: The labels of the training set.
        :return:
        """
        self._weights = [0 for index in range(len(x[0])+1)]

        for epoch in range(self._max_epochs):
            error_rate = 0

            for index in range(len(x)):
                point = x[index]
                prediction = self._predict(point)
                error = y[index] - prediction
                error_rate += error ** 2
                self._weights[0] += self._scale * error
                self._weights[1:] += self._scale * error * point
            # print(error_rate)
            if error_rate == 0:
                break

    def get_weights(self):
        """
        The weights of the perceptron. Run Perceptron.fit() first.
        :return:
        """
        return self._weights

    def predict(self, x: np.ndarray):
        """
        Classifies the given datapoint based on the trained weights from Perceptron.fit()
        :param x: The n dimensional datapoint to classify
        :return: The point's predicted class
        """
        retval = self._predict(x)
        return retval

    def plot_decision_regions(self, X, y, classifier, resolution=0.02):
        """
        Plots the regions of the perceptron based on weight.
        :param X: The data points
        :param y: The labels
        :param classifier: the classifier object
        :param resolution: the resolution of the regions.
        :return:
        """
        # setup marker generator and color map
        shape = X.shape
        if shape[1] == 2:
            markers = ('s', 'x', 'o', '^', 'v')
            colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
            cmap = ListedColormap(colors[:len(np.unique(y))])
            # plot the decision surface
            x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))
            Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            Z = Z.reshape(xx1.shape)
            plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
            plt.xlim(xx1.min(), xx1.max())
            plt.ylim(xx2.min(), xx2.max())
            # plot class samples
            for idx, cl in enumerate(np.unique(y)):
                plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                            alpha=0.8, c=cmap(idx),
                            marker=markers[idx], label=cl)

        elif shape[1] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
            cmap = ListedColormap(colors[:len(np.unique(y))])
            # plot the decision surface
            x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            x3_min, x3_max = X[:, 2].min() - 2, X[:, 2].max() + 1
            xx1, xx2, xx3 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                        np.arange(x2_min, x2_max, resolution),
                                        np.arange(x3_min, x3_max, resolution))
            l1, l2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                 np.arange(x2_min, x2_max, resolution))
            # Z = classifier.net_input(np.array([xx1.ravel(), xx2.ravel(), xx3.ravel()]).T)
            w = classifier.get_weights()
            Z = -1 * (w[0] + w[1] * l1.ravel() + w[2] * l2.ravel()) / w[3]
            Z.reshape(l1.shape)
            print(Z.shape)
            ax.plot(l1.ravel(), l2.ravel(), Z)
            ax.set_xlim(xx1.min(), xx1.max())
            ax.set_ylim(xx2.min(), xx2.max())
            ax.set_zlim(xx3.min(), xx3.max())
            ax.scatter(X[:50, 0], X[:50, 1], X[:50, 2], color='red', marker='o', label='setosa')
            ax.scatter(X[50:100, 0], X[50:100, 1], X[50:100, 2], color='blue', marker='x', label='versicolor')
            ax.legend()
