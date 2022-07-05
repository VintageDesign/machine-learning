import numpy as np


class ARMA:
    def __init__(self, p: int, q: int, max_epochs=1000, learning_rate=.001):
        self._p = p
        self._q = q
        self._ar_weights = np.zeros(p + 1)
        self._ma_weights = np.zeros(q + 1)
        self._max_epochs = max_epochs
        self._learning_rate = learning_rate

        self._residuals = []
        self._data = np.zeros(1)

    def forecast(self, future_count: int):

        results = []
        for index in range(future_count):
            start_idx = self._data.shape[0] - self._p
            prediction_data = self._data[start_idx:]

            ar = self._predict_ar(prediction_data)
            self._data = np.append(self._data, ar)
            results.append(ar)

        return results

    def _predict_ar(self, data):
        return np.dot(data, self._ar_weights[1:]) + self._ar_weights[0]

    def _predict_ma(self, data):
        if self._q > 0:
            return np.dot(data, self._ma_weights[1:]) + self._ma_weights[0]
        else:
            return 0

    def fit(self, data: np.ndarray):
        """
        Trains the ARMA.
        :param data: The time series datapoints to train on.
        :return:
        """
        self._data = data
        self._fit_AR(data)

        if self._q > 0:
            self._residuals = []
            for index in range(data.shape[0] - self._p):
                self._residuals.append(self._predict_ar(data[index:index+self._p]) - data[index+self._p])

            self._fit_MA(np.asarray(self._residuals))

    def _fit_AR(self, data: np.ndarray):

        for epoch in range(self._max_epochs):
            for index in range(data.shape[0] - self._p):
                prediction = self._predict_ar(data[index:index + self._p])

                error = prediction - data[index + self._p]

                self._ar_weights[0] = self._ar_weights[0] - (self._learning_rate * error)
                self._ar_weights[1:] = self._ar_weights[1:] - (self._learning_rate * error * data[index])

    def _fit_MA(self, data: np.ndarray):

        for epoch in range(self._max_epochs):
            for index in range(data.shape[0] - self._q):
                prediction = self._predict_ma(data[index:index + self._q])

                error = prediction - data[index + self._q]

                self._ma_weights[0] = self._ma_weights[0] - (self._learning_rate * error)
                self._ma_weights[1:] = self._ma_weights[1:] - (self._learning_rate * error * data[index])
