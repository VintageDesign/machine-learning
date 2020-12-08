import numpy as np
from .KNN import KNN


class PCA:
    def __init__(self, training_data:np.ndarray, classes:np.ndarray, neighbors: int = 2,  target_recovery: float = .7):
        self._mean_data = training_data.mean(axis=0)
        centered_data = training_data - self._mean_data

        self._U, self._S, _ = np.linalg.svd(centered_data.T)
        self._calculate_energy_recovery(target_recovery)

        self._training_points = np.matmul(centered_data, self._U[:, :self._k])
        self._data = np.zeros((self._training_points.shape[0], self._training_points.shape[1] + 1))
        self._data[:, :-1] = self._training_points
        self._data[:, -1] = classes
        self._knn = KNN(neighbors, self._data)

    def _calculate_energy_recovery(self, target_energy=.7):
        k = 0
        actual_energy = 0
        norm2 = np.sum(self._S)

        while target_energy > actual_energy:
            k += 1
            sigma_val = np.sum(self._S[:k])
            actual_energy = sigma_val / norm2

        self._k = k

    def predict(self, datapoint):
        point = datapoint - self._mean_data

        reduced = np.matmul(point, self._U[:, :self._k])
        index = int(self._knn.predict(reduced))
        return index

