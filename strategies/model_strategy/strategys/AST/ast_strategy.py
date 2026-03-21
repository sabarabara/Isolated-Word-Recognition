"""AST モデルストラテジー。"""
from pathlib import Path
from typing import Optional

from strategies.registry import MODEL_REGISTRY
from strategies.model_strategy.model_strategy import ModelStrategy


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

        annotation_dir = Path(self.config.get("annotation_dir", "data/annotation_data"))
        audio_dir = Path(self.config.get("audio_dir", "data/outputs/segment"))
        batch_size = int(self.config.get("batch_size", 8))

        dataset = ASTDataset(annotation_dir, audio_dir, config=self.config)

        from torch.utils.data import DataLoader, random_split
        n_total = len(dataset)
        n_val = max(1, int(0.2 * n_total))
        n_train = n_total - n_val
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        print(f"prepare_dataloader: done (train={n_train}, val={n_val})")

    def build(self):
        try:
            _, ASTModel, _, _ = self._lazy_imports()
        except Exception as e:
            print(f"build: skipped (missing deps): {e}")
            return

        import torch
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

        if self.model is None or self.train_loader is None:
            print("train: skipped (model or dataloader missing)")
            return

        trainer = Trainer(
            model=self.model,
            config=self._make_exp_config(),
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            device=self.device,
            output_dir=Path(self.config.get("output_dir", ".")),
        )
        self.trainer = trainer
        trainer.train()
        print("train: completed")

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
