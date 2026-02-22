from typing import List, Optional

from strategies.evaluation_strategy.evaluation_strategy import EvaluationStrategy


class CompositeEvaluation(EvaluationStrategy):
    def __init__(self, strategies: Optional[List[EvaluationStrategy]] = None):
        self.strategies = strategies or []

    def evaluate(self, output=None, target=None) -> dict:
        combined_results = {}
        for strategy in self.strategies:
            try:
                combined_results.update(strategy.evaluate(output, target))
            except Exception:
                # ignore failures from individual strategies at evaluation time
                pass
        return combined_results
