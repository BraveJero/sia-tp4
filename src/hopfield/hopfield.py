import numpy as np


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        pattern_size = len(patterns[0])

        if self.n_neurons * 0.15 < len(patterns):
            raise RuntimeWarning("The maximum number of patterns you can store is equal to 15% of the number of "
                                 "network neurons.")

        for pattern in patterns:
            pattern = np.array(pattern).reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)

        self.weights /= pattern_size

    def recall(self, pattern, max_iter=100):
        pattern = np.array(pattern).reshape(-1, 1)
        for _ in range(max_iter):
            updated_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(updated_pattern, pattern):
                return updated_pattern.flatten()
            pattern = updated_pattern
        return pattern.flatten()
