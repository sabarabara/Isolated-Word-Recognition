from strategies.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from strategies.registry import EVAL_REGISTRY


@EVAL_REGISTRY.register("mock")
class SimpleEvalStrategy(EvaluationStrategy):
    """Minimal evaluation strategy used for testing and fallback.

    Returns a constant dummy metric.
    """

    def evaluate(self, output=None, target=None) -> dict:
        return {"mock_eval": 0.0}
