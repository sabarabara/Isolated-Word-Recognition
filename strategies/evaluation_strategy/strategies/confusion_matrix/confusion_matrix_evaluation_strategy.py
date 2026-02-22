import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from strategies.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from .registry import EVAL_REGISTRY


@EVAL_REGISTRY.register("confusion")
class ConfusionMatrixStrategy(EvaluationStrategy):
    def __init__(self, num_classes=100):
        self.num_classes = num_classes

    def evaluate(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        preds = output.argmax(dim=1)

        y_true = target.detach().cpu().numpy()
        y_pred = preds.detach().cpu().numpy()

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.num_classes))

        return {"confusion_matrix": cm}
