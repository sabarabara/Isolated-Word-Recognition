from pathlib import Path
from typing import Optional

from strategies.registry import MODEL_REGISTRY
from strategies.model_strategy.model_strategy import ModelStrategy
from strategies.model_strategy.strategys._shared.dataloader_builder import (
    build_train_val_loaders,
)
from strategies.model_strategy.strategys._shared.train_runner import run_training
from preprocess.augmentation import build_audio_augmentation


@MODEL_REGISTRY.register("cnn2d")
class CNN2DStrategy(ModelStrategy):
    """Glue code to wire CNN2D dataset, model, trainer and evaluator.

    This strategy lazily imports heavy dependencies to avoid import-time
    failures when torch is not installed. It provides the same interface
    as other ModelStrategy implementations used by the project.
    """

    def __init__(self, eval_strategy: Optional[object] = None, config: dict = None):
        super().__init__(eval_strategy=eval_strategy)
        self.config = config or {}
        self.device = None
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None

    def _lazy_imports(self):
        # local imports so missing torch or librosa won't break module import
        from strategies.model_strategy.strategys.cnn2d.dataset import CNN2DDataset
        from strategies.model_strategy.strategys.cnn2d.model import CNN2DModel
        from strategies.model_strategy.strategys.cnn2d.trainer import Trainer
        from strategies.model_strategy.strategys.cnn2d.evaluator import Evaluator

        return CNN2DDataset, CNN2DModel, Trainer, Evaluator

    def _make_exp_config(self):
        from configs.experiment_config import make_experiment_config

        return make_experiment_config(self.config)

    def prepare_dataloader(self):
        try:
            CNN2DDataset, _, _, _ = self._lazy_imports()
        except Exception as e:
            print(f"prepare_dataloader: skipped (missing deps): {e}")
            return

        annotation_dir = Path(self.config.get("annotation_dir", "data/annotation_data"))
        audio_dir = Path(self.config.get("audio_dir", "data/outputs/segment"))

        batch_size = int(self.config.get("batch_size", 8))

        train_aug = build_audio_augmentation(self.config)
        self.train_loader, self.val_loader, n_train, n_val = build_train_val_loaders(
            CNN2DDataset,
            annotation_dir,
            audio_dir,
            config=self.config,
            batch_size=batch_size,
            train_augmentation=train_aug,
        )

        print(f"prepare_dataloader: done (train={n_train}, val={n_val})")

    def build(self):
        try:
            _, CNN2DModel, _, _ = self._lazy_imports()
        except Exception as e:
            print(f"build: skipped (missing deps): {e}")
            return

        import torch

        device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device = device

        exp_cfg = self._make_exp_config()
        self.model = CNN2DModel(exp_cfg.model)
        print("build: model constructed")

    def train(self):
        try:
            _, _, Trainer, _ = self._lazy_imports()
        except Exception as e:
            print(f"train: skipped (missing deps): {e}")
            return
        return run_training(self, Trainer)

    def evaluate(self, output=None, target=None):
        try:
            _, _, _, Evaluator = self._lazy_imports()
        except Exception as e:
            print(f"evaluate: skipped (missing deps): {e}")
            return {}

        if self.model is None or self.val_loader is None:
            print("evaluate: skipped (model or val_loader missing)")
            return {}

        evaluator = Evaluator(
            model=self.model,
            config=self._make_exp_config(),
            device=self.device,
            output_dir=Path(self.config.get("output_dir", ".")),
        )

        metrics = evaluator.evaluate(self.val_loader, save_predictions=False)
        if self.eval_strategy:
            try:
                # Allow composed eval strategy to combine results
                extra = self.eval_strategy.evaluate(metrics, None)
                if isinstance(extra, dict):
                    metrics.update(extra)
            except Exception:
                pass

        return metrics
