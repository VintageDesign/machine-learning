import numpy as np


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

