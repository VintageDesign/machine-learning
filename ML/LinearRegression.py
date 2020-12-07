import numpy as np
from math import sqrt


class LinearRegression:
    def __init__(self):
        self._slope = 0
        self._intercept = 0
        self._interval = 0

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

        top = np.dot(x - x_mean, y - y_mean)
        bottom = np.sum(((x - x_mean) ** 2))
        self._slope = top / bottom
        self._intercept = y_mean - (self._slope * x_mean)

        y_hat = self._slope * x + self._intercept
        err = np.sum((y - y_hat)**2)
        deviation = sqrt(1 / (len(y) - 2) * err)

        self._interval = 1.96 * deviation

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

    def get_interval(self):
        return self._interval
