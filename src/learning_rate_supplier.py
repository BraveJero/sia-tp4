from abc import ABC
from numbers import Number


class LearningRateSupplier(ABC):
    def supply(self) -> Number:
        raise NotImplementedError


class ConstantLearningRate(LearningRateSupplier):
    def __init__(self, learning_rate: Number):
        self._learning_rate = learning_rate

    def supply(self) -> Number:
        return self._learning_rate


class ExponentialDecayLearningRate(LearningRateSupplier):
    def __init__(self, learning_rate: Number, decay: Number):
        self._learning_rate = learning_rate
        self._decay = decay

    def supply(self) -> Number:
        ans = self._learning_rate
        self._learning_rate *= self._decay
        return ans
