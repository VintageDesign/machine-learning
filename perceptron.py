import numpy as np


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
            print(error_rate)
            if error_rate == 0:
                break

    def get_weights(self):
        return self._weights

    def predict(self, x):
        size = x.shape
        size = size[1]
        retval = self._predict(x)
        return retval

