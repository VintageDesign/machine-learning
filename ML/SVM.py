import numpy as np


class SVM:
    def __init__(self, learning_rate: float, epochs: int, margin: float):
        self._learning_rate = learning_rate
        self._weights = np.zeros((1, 1))
        self._epochs = epochs
        self._margin = margin

    def predict(self, points: np.ndarray):
        """
        :param points:
        :return:
        """
        if len(points.shape) == 1:
            return np.sign(np.dot(self._weights[1:], points[:-1]) + self._weights[0])
        else:
            return np.sign(np.dot(self._weights[1:], points[:, :-1]) + self._weights[0])

    def fit(self, dataset: np.ndarray):
        """
        Fit the SVM using SGD
        :param dataset: the ndarray of data where dataset[:, -1] is the class of the data point
        :return:
        """
        dataset = dataset.astype(np.float64)
        self._weights = np.zeros(dataset.shape[1])

        for _ in range(self._epochs):
            for _, point in enumerate(dataset):
                predicted_val = self.predict(point)
                error = point[-1] * predicted_val < 1
                if not error:
                    self._weights[1:] -= self._learning_rate * 1/2 * self._margin * self._weights[1:]
                else:
                    self._weights[1:] -= self._learning_rate * 1/2 * self._margin * self._weights[1:] - \
                                         np.dot(point[:-1], point[-1])
                    self._weights[0] -= self._learning_rate * point[-1]

    def get_weights(self):
        return self._weights


