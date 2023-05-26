import numpy as np


class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        pattern_size = len(patterns[0])

        # if self.n_neurons * 0.15 < len(patterns):  # TODO: Check this.
        #     raise RuntimeWarning(f"The maximum number of patterns you can store is equal to 15% of the number of "
        #                          f"network neurons. Number of patterns = {len(patterns)} !<= Number of neurons * 0.15"
        #                          f" =  {self.n_neurons * 0.15}")

        for pattern in patterns:
            pattern = np.array(pattern).reshape(-1, 1)
            self.weights += np.dot(pattern, pattern.T)
        np.fill_diagonal(self.weights, 0)

        self.weights /= pattern_size

    def recall(self, pattern, max_iter=100):
        pattern = np.array(pattern).reshape(-1, 1)
        pattern_history = [pattern]
        for _ in range(max_iter):
            updated_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(updated_pattern, pattern):
                self.energy(updated_pattern)
                return pattern_history, True
            pattern = updated_pattern
            pattern_history.append(pattern)
        return pattern_history, False

    def energy(self, pattern):
        return - 0.5 * np.dot(pattern.flatten(), np.dot(self.weights, pattern).flatten())
