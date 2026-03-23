"""Conformer-like model strategy (CNN + Transformer encoder)."""

from pathlib import Path
from typing import Optional

import torch

from strategies.registry import MODEL_REGISTRY
from strategies.model_strategy.model_strategy import ModelStrategy
from strategies.model_strategy.strategys._shared.dataloader_builder import (
    build_train_val_loaders,
)
from strategies.model_strategy.strategys._shared.lazy_imports import StrategyLazyImports
from strategies.model_strategy.strategys._shared.train_runner import run_training


@MODEL_REGISTRY.register("conformer")
class ConformerStrategy(ModelStrategy):
    def __init__(self, eval_strategy: Optional[object] = None, config: dict = None):
        super().__init__(eval_strategy=eval_strategy)
        self.config = config or {}
        self.device = None
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None

    def _lazy_imports(self):
        from strategies.model_strategy.strategys.conformer.dataset import ConformerDataset
        from strategies.model_strategy.strategys.conformer.model import ConformerModel
        from strategies.model_strategy.strategys._shared.base_trainer import Trainer
        from strategies.model_strategy.strategys._shared.base_evaluator import Evaluator

        return StrategyLazyImports(
            dataset_cls=ConformerDataset,
            model_cls=ConformerModel,
            trainer_cls=Trainer,
            evaluator_cls=Evaluator,
            default_cam_layer="cnn_frontend.net.10",
        )

    def _make_exp_config(self):
        from configs.experiment_config import make_experiment_config

        return make_experiment_config(self.config)

    def prepare_dataloader(self):
        try:
            ConformerDataset, _, _, _ = self._lazy_imports()
        except Exception as e:
            print(f"prepare_dataloader: skipped (missing deps): {e}")
            return

        audio_dir = Path(self.config.get("audio_dir", "data/outputs/segment"))
        batch_size = int(self.config.get("batch_size", 8))

        self.train_loader, self.val_loader, n_train, n_val = build_train_val_loaders(
            ConformerDataset,
            audio_dir,
            config=self.config,
            batch_size=batch_size,
        )
        print(f"prepare_dataloader: done (train={n_train}, val={n_val})")

    def build(self):
        try:
            _, ConformerModel, _, _ = self._lazy_imports()
        except Exception as e:
            print(f"build: skipped (missing deps): {e}")
            return

        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = ConformerModel(config=self.config)
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
                extra = self.eval_strategy.evaluate(metrics, None)
                if isinstance(extra, dict):
                    metrics.update(extra)
            except Exception:
                pass
        return metrics
