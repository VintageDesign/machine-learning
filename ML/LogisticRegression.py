from math import exp
import numpy as np


class LogisticRegression:
    """
    Fits a logistic curve to the given data set.
    """

    def __init__(self, l_rate, n_epoch):
        self._l_rate = l_rate
        self._n_epoch = n_epoch
        self._coeff = np.zeros((1, 1))

    def predict(self, row):
        yhat = self._coeff[0] + self._coeff[1] * row
        return 1.0 / (1.0 + np.exp(-yhat))

    def fit(self, train, classes):
        try:
            self._coeff = [0.0 for i in range(train.shape[1] + 1)]
        except IndexError:
            self._coeff = [0.0 for i in range(2)]

        for epoch in range(self._n_epoch):
            sum_error = 0
            for i in range(train.shape[0]):
                row = train[i]
                yhat = self.predict(row)
                error = classes[i] - yhat
                sum_error += error ** 2
                self._coeff[0] = self._coeff[0] + self._l_rate * error * yhat * (1.0 - yhat)
                for i in range(1):
                    self._coeff[i + 1] = self._coeff[i + 1] + self._l_rate * error * yhat * (1.0 - yhat) * row
                if sum_error == 0:
                    break

            if sum_error == 0:
                break

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, self._l_rate, sum_error))

    def get_coeff(self):
        return self._coeff
