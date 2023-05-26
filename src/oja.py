from numbers import Number

import numpy as np


class OjaRule:
    @staticmethod
    def initialize_weights(n: int) -> np.ndarray:
        return np.random.uniform(0, 1, n)
