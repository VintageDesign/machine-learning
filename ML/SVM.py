import numpy as np


class SVM:
    def __init__(self, learning_rate: float, epochs: int, margin: float):
        self._learning_rate = learning_rate
        self._weights = np.zeros((1, 1))
        self._epochs = epochs
        self._margin = margin

    def _compute_cost(self, points):
        distances = 1 - points[:, -1] * (np.dot(points[:, :-1], self._weights[1:]) + self._weights[0])
        distances[distances < 0] = 0
        loss = self._margin * np.mean(distances)

        return 1/2 * np.dot(self._weights, self._weights) + loss

    def _compute_cost_gradient(self, points):

        distance = 1 - (points[-1] * (np.dot(points[:-1], self._weights[1:]) + self._weights[0]))
        delta_weights = np.zeros(self._weights.shape).astype(np.float64)

        if distance < 0:
            delta_weights += self._weights
        else:
            delta_weights[1:] += self._weights[1:] - (self._margin * points[-1] * points[:-1]).astype(np.float64)
            delta_weights[0] += self._weights[0] - (self._margin * points[-1])

        return delta_weights

    def fit(self, dataset):
        last_cost = float("inf")
        self._weights = np.zeros(dataset.shape[1]).astype(np.float64)

        for _ in range(self._epochs):
            for ind, point in enumerate(dataset):
                ascent = self._compute_cost_gradient(point)
                self._weights = self._weights - (self._learning_rate * ascent)
            cost = self._compute_cost(dataset)
            last_cost = cost


    def get_weights(self):
        return self._weights

