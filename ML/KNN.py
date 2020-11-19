import numpy as np


class KNN:
    def __init__(self, k, training_set):
        self._k = k
        self._training_set = training_set

    def predict(self, query):
        tmp = (self._training_set[:, :-1] - query).astype(float)
        distances = np.linalg.norm(tmp, axis=1)

        index = np.argsort(distances)

        sorted_set = self._training_set[index[:], :]

        (unique, counts) = np.unique(sorted_set[:self._k, -1], return_counts=True)

        return unique[counts == np.max(counts)][0]

