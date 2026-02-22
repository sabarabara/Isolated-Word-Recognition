from abc import ABC, abstractmethod


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, output=None, target=None) -> dict:
        raise NotImplementedError()
