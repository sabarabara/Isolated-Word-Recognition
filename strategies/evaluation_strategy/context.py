class EvaluationContext:
    def __init__(self, strategies: list):
        self.strategies = strategies

    def run(self, output, target):
        results = {}
        for strategy in self.strategies:
            res = strategy.evaluate(output, target)
            if res:
                results.update(res)
        return results
