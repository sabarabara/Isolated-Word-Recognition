from strategies.model_strategy.model_strategy import ModelStrategy


class ModelContext:
    def __init__(self, strategy: ModelStrategy):
        self.strategy = strategy

    def run_full_pipeline(self):
        self.strategy.prepare_dataloader()
        self.strategy.build()
        output, target = self.strategy.train()
        return self.strategy.evaluate(output, target)

    def prepare(self):
        self.strategy.prepare_dataloader()

    def build(self):
        self.strategy.build()

    def train(self):
        self.strategy.train()

    def evaluate(self, *args, **kwargs):
        return self.strategy.evaluate(*args, **kwargs)
