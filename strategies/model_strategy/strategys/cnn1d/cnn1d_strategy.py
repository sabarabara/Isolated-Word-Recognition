from typing import Optional
from strategies.model_strategy.model_strategy import ModelStrategy
from strategies.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("1dcnn")
class CNN1DStrategy(ModelStrategy):
    def __init__(
        self, eval_strategy: Optional[object] = None, config: dict = None, model=None
    ):
        super().__init__(eval_strategy)
        self.config = config or {}
        self.model = model
        self.trainer = None
        self.evaluator = None
        self.train_loader = None
        self.val_loader = None

    def prepare_dataloader(self):
        # perform lazy imports so this module can be imported even without torch
        try:
            from strategies.model_strategy.strategys.cnn1d.dataset import (
                RandomAudioDataset,
            )
            from torch.utils.data import DataLoader
        except Exception as e:
            raise RuntimeError("Required dataset dependencies missing: " + str(e))

        num_classes = self.config.get("num_classes", 100)
        train_ds = RandomAudioDataset(
            num_samples=128,
            length=self.config.get("input_length", 16000),
            num_classes=num_classes,
        )
        val_ds = RandomAudioDataset(
            num_samples=32,
            length=self.config.get("input_length", 16000),
            num_classes=num_classes,
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=self.config.get("batch_size", 8), shuffle=True
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config.get("batch_size", 8), shuffle=False
        )

    def build(self):
        num_classes = self.config.get("num_classes", 100)
        in_channels = self.config.get("in_channels", 1)
        # lazy-import model/trainer/evaluator so module import doesn't require torch
        try:
            from strategies.model_strategy.strategys.cnn1d.model import CNN1DModel
            from strategies.model_strategy.strategys.cnn1d.trainer import Trainer
            from strategies.model_strategy.strategys.cnn1d.evaluator import Evaluator
        except Exception as e:
            raise RuntimeError(
                "Required model/training dependencies missing: " + str(e)
            )

        # if a model instance was provided by factories, use it; otherwise build one
        if self.model is None:
            self.model = CNN1DModel(in_channels=in_channels, num_classes=num_classes)
        self.trainer = Trainer(self.model, lr=self.config.get("lr", 1e-3))
        self.evaluator = Evaluator(self.model)

    def train(self):
        epochs = self.config.get("epochs", 1)
        if self.train_loader is None or self.trainer is None:
            raise RuntimeError(
                "Dataloader or trainer not prepared. Call prepare_dataloader() and build() first."
            )
        self.trainer.fit(self.train_loader, epochs=epochs, val_loader=self.val_loader)

    def evaluate(self, output=None, target=None):
        if self.val_loader is None or self.evaluator is None:
            return {}
        return self.evaluator.evaluate(self.val_loader)


# Register this strategy and model with the central context registry if available
try:
    from strategies.model_strategy.context import register_strategy

    # try to import the concrete model class for registration; ignore if missing
    try:
        from strategies.model_strategy.strategys.cnn1d.model import (
            CNN1DModel as _ModelCls,
        )
    except Exception:
        _ModelCls = None
    register_strategy("1dcnn", strategy_cls=CNN1DStrategy, model_cls=_ModelCls)
except Exception:
    # registration is optional (e.g., when imports fail during static analysis)
    pass
