import numpy as np
import matplotlib.pyplot as plt
from math import inf
from matplotlib.colors import ListedColormap


class Perceptron:
    def __init__(self, scale, max_epochs):
        self._scale = scale
        self._max_epochs = max_epochs
        self._weights = []

    def _predict(self, x):
        summation = self.net_input(x)
        summation = np.where(summation > 0, 1, -1)
        return summation

    def net_input(self, x):
        return np.dot(x, self._weights[1:]) + self._weights[0]

    def fit(self, x, y):
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
        return self._weights

    def predict(self, x):
        retval = self._predict(x)
        return retval

    def plot_decision_regions(self, X, y, classifier, resolution=0.02):
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


class LinearRegression:
    def __init__(self):
        self._slope = 0
        self._intercept = 0

    def _get_mean(self, arr):
        """
        Calculates the mean of the array given

        :param arr: The given array
        :return: Mean
        """

        return np.mean(arr)

    def fit(self, x, y):
        """
        Fits a linear model using least squares.

        :param x: The list of independent variables
        :param y: The list of dependent variables
        :return: bool success
        """
        if len(x) != len(y):
            print("Error: input list sizes must agree.")
            raise AttributeError
        x_mean = self._get_mean(x)
        y_mean = self._get_mean(y)

        top = 0
        bottom = 0
        for index in range(len(x)):
            top += (x[index] - x_mean) * (y[index] - y_mean)
            bottom += (x[index] - x_mean) ** 2

        self._slope = top / bottom
        self._intercept = y_mean - (self._slope * x_mean)

        return True

    def get_slope(self):
        """
        :return: The slope of the fit line
        """
        return self._slope

    def get_intercept(self):
        """
        :return: The intercept of the fit line.
        """
        return self._intercept


class Stump:
    def __init__(self):
        self._weights = np.ones((5, 1))/5
        self._steps = 10

        self._threshold = 0
        self._ineq = ''
        self._minerror = inf
        self._dim = 0

    def _classify(self, data_matrix, dimension, threshold, inequality):
        """
        Classifies every data point based on the threshold
        :param data_matrix: the datapoints
        :param dimension: the dimension to classify in
        :param threshold: the threshold
        :param inequality: tells us if we are classifying points above or below the threshold
        :return: The classification of each point
        """
        retval = np.ones((data_matrix.shape[0], 1))

        if inequality == 'lt':
            retval[data_matrix[:, dimension] <= threshold] = -1
        else:
            retval[data_matrix[:, dimension] > threshold] = -1

        return retval

    def fit(self, data, labels):
        """
        Fits a Decision Stump to the data
        :param data: the data to fit
        :param labels: the classification labels for the data.
        :return:
        """
        data_matrix = np.mat(data)
        label_matrix = np.mat(labels).T

        m, n = data_matrix.shape

        for i in range(n):
            min = data_matrix[:, i].min()
            max = data_matrix[:, i].max()

            step_size = (max - min) / self._steps

            for j in range(-1, int(self._steps) + 1):
                for inequal in ['lt', 'gt']:

                    threshold = min + j * step_size
                    predicted_val = self._classify(data_matrix, i, threshold, inequal)

                    error = np.ones((m, 1))

                    error[predicted_val == label_matrix] = 0

                    error_total = np.sum(error)

                    if error_total < self._minerror:
                        self._minerror = error_total
                        self._threshold = threshold
                        self._dim = i
                    self._ineq = inequal

    def get_threshold(self):
        """
        :return: The fit threshold
        """
        return self._threshold

    def get_dimension(self):
        """
        :return: the fit dimension
        """
        return self._dim