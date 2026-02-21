from abc import ABC, abstractmethod


class ModelStrategy(ABC):
    def __init__(self, eval_strategy: EvaluationStrategy):
        self.eval_strategy = eval_strategy

    @abstractmethod
    def prepare_dataloader(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        return self.eval_strategy.evaluate()
