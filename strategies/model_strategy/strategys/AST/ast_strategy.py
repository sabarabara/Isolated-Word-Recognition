"""AST モデルストラテジー。"""

from pathlib import Path
from typing import Optional
import torch

from strategies.registry import MODEL_REGISTRY
from strategies.model_strategy.model_strategy import ModelStrategy
from strategies.model_strategy.strategys._shared.dataloader_builder import (
    build_train_val_loaders,
)
from strategies.model_strategy.strategys._shared.train_runner import run_training
from preprocess.augmentation import build_audio_augmentation


@MODEL_REGISTRY.register("ast")
class ASTStrategy(ModelStrategy):
    def __init__(self, eval_strategy: Optional[object] = None, config: dict = None):
        super().__init__(eval_strategy=eval_strategy)
        self.config = config or {}
        self.device = None
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None

    def _lazy_imports(self):
        from strategies.model_strategy.strategys.AST.dataset import ASTDataset
        from strategies.model_strategy.strategys.AST.model import ASTModel
        from strategies.model_strategy.strategys._shared.base_trainer import Trainer
        from strategies.model_strategy.strategys._shared.base_evaluator import Evaluator

        return ASTDataset, ASTModel, Trainer, Evaluator

    def _make_exp_config(self):
        from configs.experiment_config import make_experiment_config

        return make_experiment_config(self.config)

    def prepare_dataloader(self):
        try:
            ASTDataset, _, _, _ = self._lazy_imports()
        except Exception as e:
            print(f"prepare_dataloader: skipped (missing deps): {e}")
            return

        annotation_dir = Path(self.config.get("annotation_dir"))
        assert annotation_dir is not None, (
            "annotation_dir must be specified in config ast_strategy.py"
        )
        audio_dir = Path(self.config.get("audio_dir"))
        assert audio_dir is not None, (
            "audio_dir must be specified in config ast_strategy.py"
        )
        batch_size = int(self.config.get("batch_size"))
        assert batch_size > 0, (
            "batch_size must be a positive integer in config ast_strategy.py"
        )

        train_aug = build_audio_augmentation(self.config)
        self.train_loader, self.val_loader, n_train, n_val = build_train_val_loaders(
            ASTDataset,
            annotation_dir,
            audio_dir,
            config=self.config,
            batch_size=batch_size,
            train_augmentation=train_aug,
        )
        print(f"prepare_dataloader: done (train={n_train}, val={n_val})")

    def build(self):
        try:
            _, ASTModel, _, _ = self._lazy_imports()
        except Exception as e:
            print(f"build: skipped (missing deps): {e}")
            return

        self.device = torch.device(
            self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = ASTModel(config=self.config)
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
