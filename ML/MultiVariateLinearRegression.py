import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap



class MultiVariateLinearRegression:
    def __init__(self, learning_rate: float, max_epochs: int):
        """
        Initializes the Regression  object
        :param scale: The scale at which the weights of the classifier are manipulated.
        :param max_epochs: The maximum number of times the fit function will run if the classifier cannot converge.
        """
        self._learning_rate = learning_rate
        self._max_epochs = max_epochs
        self._weights = np.zeros((1,1))

    def predict(self, x):
        summation = self._net_input(x)
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
        Trains the linear regression.
        :param x: The datapoints to train on.
        :param y: The labels of the training set.
        :return:
        """
        try:
            self._n = x.shape[0]
            num_coeffs = x.shape[1] + 1
        except IndexError:
            num_coeffs = 2

        self._weights = np.zeros(num_coeffs)

        for epoch in range(self._max_epochs):
            for index in range(self._n):
                prediction = self.predict(x[index])

                error = prediction - y[index]

                self._weights[0] = self._weights[0] - (self._learning_rate * error)
                self._weights[1:] = self._weights[1:] - (self._learning_rate * error * x[index])

    def get_weights(self):
        """
        The weights of the function, first value in the array is the bias.
        :return:
        """
        return self._weights
