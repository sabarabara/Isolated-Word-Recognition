from abc import ABC, abstractmethod
from strategies.evaluation_strategy.evaluation_strategy import EvaluationStrategy


class ModelStrategy(ABC):
    def __init__(self, eval_strategy: EvaluationStrategy):
        self.eval_strategy = eval_strategy

    @abstractmethod
    def prepare_dataloader(self):
        raise NotImplementedError()

    @abstractmethod
    def build(self):
        raise NotImplementedError()

    @abstractmethod
    def train(self):
        raise NotImplementedError()

    def evaluate(self, output=None, target=None):
        if self.eval_strategy is None:
            return {}
        if output is None or target is None:
            return {}
        return self.eval_strategy.evaluate(output, target)
