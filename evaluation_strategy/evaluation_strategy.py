from abc import ABC, abstractmethod

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class EvaluationStrategy(ABC):
    @abstractmethod
    def evaluate(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        pass
