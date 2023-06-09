import numpy as np

from src.oja.learning_rate_supplier import LearningRateSupplier


class OjaRule:
    @staticmethod
    def initialize_weights(n: int) -> np.ndarray:
        return np.random.uniform(0, 1, n)

    @staticmethod
    def train(data: np.ndarray,
              learning_rate_supplier: LearningRateSupplier,
              weights: np.ndarray,
              epochs: int):
        for epoch in range(epochs):
            learning_rate = learning_rate_supplier.supply()
            for data_input in data:
                data_output = np.dot(data_input, weights)
                dw = learning_rate * data_output * (data_input - data_output * weights)
                weights += dw

    @staticmethod
    def test(data: np.ndarray,
             weights: np.ndarray):
        return np.dot(data, weights)
