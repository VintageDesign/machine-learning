from math import exp
import numpy as np

class LogisticRegression:
    """
    Fits a logistic curve to the given data set.
    """

    def __init__(self):
        self._coeff = np.zeros((1, 1))

    def predict(self, row):
        yhat = self._coeff[0]
        for i in range(len(row) - 1):
            yhat += self._coeff[i + 1] * row[i]
        return 1.0 / (1.0 + exp(-yhat))

    def coefficients_sgd(self, train, l_rate, n_epoch):
        self._coeff = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                yhat = self.predict(row)
                error = row[-1] - yhat
                sum_error += error ** 2
                self._coeff[0] = self._coeff[0] + l_rate * error * yhat * (1.0 - yhat)
                for i in range(len(row) - 1):
                    self._coeff[i + 1] = self._coeff[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        return self._coeff