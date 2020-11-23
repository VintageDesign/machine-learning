import numpy as np


class KNN:
    def __init__(self, k: int, training_set: np.ndarray):
        """
        Initializes the KNN object.
        :param k: The number of neighbors to compare the query to.
        :param training_set: The training set of datapoints where datapoint[-1] is the class of the datapoint.
        """
        self._k = k
        self._training_set = training_set

    def predict(self, query: np.ndarray):
        """
        Predicts the class of the query datapoint.
        :param query: the datapoint to query must match the shape of the data points in the training set.
        :return:
        """
        assert query.shape == self._training_set[1, :-1].shape, "Size of the query does not match the size of the" \
                                                                " training set, Which is: "\
                                                                + str(self._training_set[1, :-1].shape)
        tmp = (self._training_set[:, :-1] - query).astype(float)
        distances = np.linalg.norm(tmp, axis=1)

        index = np.argsort(distances)
        sorted_set = self._training_set[index, :]

        (unique, counts) = np.unique(sorted_set[:self._k, -1], return_counts=True)

        return unique[counts == np.max(counts)][0]

