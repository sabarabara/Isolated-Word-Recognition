"""全モデル共通 Evaluator。

(inputs, labels, metadata) の 3-tuple バッチに対応。
CompositeEvaluation (TopK + ConfusionMatrix) を使用する。
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
import json
import logging

from configs.experiment_config import ExperimentConfig
from strategies.evaluation_strategy.composite import CompositeEvaluation
from strategies.evaluation_strategy.strategies.topk.topk_evaluation_strategy import (
    TopKEvaluationStrategy,
)
from strategies.evaluation_strategy.strategies.confusion_matrix.confusion_matrix_evaluation_strategy import (
    ConfusionMatrixStrategy,
)

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.output_dir = output_dir

    def evaluate(self, data_loader: DataLoader, save_predictions: bool = True) -> Dict:
        self.model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _metadata in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                all_outputs.append(outputs.cpu())
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        eval_strategy = CompositeEvaluation(
            [
                TopKEvaluationStrategy(k_list=[1, 3, 5]),
                ConfusionMatrixStrategy(num_classes=self.config.model.num_classes),
            ]
        )
        metrics = eval_strategy.evaluate(all_outputs, all_labels)

        if save_predictions:
            self._save_metrics(metrics)

        return metrics

    def _save_metrics(self, metrics: Dict):
        metrics_path = self.output_dir / "metrics" / "evaluation_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for k, v in metrics.items():
            if k == "confusion_matrix":
                serializable[k] = v.tolist() if hasattr(v, "tolist") else v
            else:
                try:
                    serializable[k] = float(v)
                except Exception:
                    serializable[k] = v
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.info(f"評価メトリクスを保存: {metrics_path}")
