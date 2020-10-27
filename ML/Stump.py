import numpy as np
from math import inf


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
