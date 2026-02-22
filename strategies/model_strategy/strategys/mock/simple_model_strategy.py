from typing import Optional
from strategies.model_strategy.model_strategy import ModelStrategy
from strategies.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("mock")
class SimpleModelStrategy(ModelStrategy):
    """Minimal model strategy used for testing and fallback.

    Delegates evaluation to attached evaluation strategy when available.
    """

    def __init__(self, eval_strategy: Optional[object] = None, config: dict = None):
        super().__init__(eval_strategy=eval_strategy)
        self.config = config or {}

    def prepare_dataloader(self):
        print("prepare_dataloader: noop (mock)")

    def build(self):
        print("build: noop (mock)")

    def train(self):
        print("train: noop (mock)")

    def evaluate(self, output=None, target=None):
        if self.eval_strategy is not None:
            try:
                return self.eval_strategy.evaluate(output, target)
            except Exception:
                return {}
        return {}
