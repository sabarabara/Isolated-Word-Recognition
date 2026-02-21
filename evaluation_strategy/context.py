class CompositeEvaluation(EvaluationStrategy):
    def __init__(self, strategies: list[EvaluationStrategy]):
        self.strategies = strategies

    def evaluate(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        combined_results = {}
        for strategy in self.strategies:
            combined_results.update(strategy.evaluate(output, target))
        return combined_results
