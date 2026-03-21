import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict
import json
import logging

from configs.experiment_config import ExperimentConfig
from strategies.evaluation_strategy.composite import CompositeEvaluation
from strategies.evaluation_strategy.strategies.topk.topk_evaluation_strategy import TopKEvaluationStrategy
from strategies.evaluation_strategy.strategies.confusion_matrix.confusion_matrix_evaluation_strategy import ConfusionMatrixStrategy

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

    def evaluate(
        self,
        data_loader: DataLoader,
        save_predictions: bool = True
    ) -> Dict:
    
        self.model.eval()
        
        all_outputs = []
        all_labels = []
        all_metadata = []

        with torch.no_grad():
            for inputs, labels, metadata in data_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                outputs = self.model(inputs)
                
                all_outputs.append(outputs.cpu())
                all_labels.append(labels)
                all_metadata.extend(metadata)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        eval_strategy = CompositeEvaluation([
            TopKEvaluationStrategy(k_list=[1, 3, 5]),
            ConfusionMatrixStrategy(num_classes=self.config.model.num_classes),
        ])
        metrics = eval_strategy.evaluate(all_outputs, all_labels)

        # 予測結果の保存
        if save_predictions:
            self._save_predictions(
                all_outputs, all_labels, all_metadata
            )

        # メトリクスの保存
        self._save_metrics(metrics)

        return metrics

    def _save_predictions(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        metadata: list
    ):
        """予測結果を保存する。"""
        predictions_path = self.output_dir / "metrics" / "predictions.json"
        
        probs = torch.softmax(outputs, dim=1)
        _, preds = outputs.max(dim=1)

        predictions_data = []
        for i in range(len(labels)):
            predictions_data.append({
                "true_label": int(labels[i].item()),
                "predicted_label": int(preds[i].item()),
                "confidence": float(probs[i, preds[i]].item()),
                "top5_labels": [int(x) for x in outputs[i].topk(5)[1].tolist()],
                "top5_probs": [float(x) for x in probs[i].topk(5)[0].tolist()],
                "metadata": {
                    k: v[i] if isinstance(v, list) else v
                    for k, v in metadata.items()
                } if isinstance(metadata, dict) else {}
            })

        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(predictions_data, f, indent=2, ensure_ascii=False)

        logger.info(f"予測結果を保存: {predictions_path}")

    def _save_metrics(self, metrics: Dict):
        """メトリクスを保存する。"""
        metrics_path = self.output_dir / "metrics" / "evaluation_metrics.json"
        
        # NumPy配列をリストに変換
        serializable_metrics = {}
        for key, value in metrics.items():
            if key == "confusion_matrix":
                serializable_metrics[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_metrics[key] = value
            else:
                serializable_metrics[key] = float(value) if value is not None else None

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

        logger.info(f"評価メトリクスを保存: {metrics_path}")