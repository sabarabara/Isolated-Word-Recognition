import torch
from strategies.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from strategies.registry import EVAL_REGISTRY


@EVAL_REGISTRY.register("topk")
class TopKEvaluationStrategy(EvaluationStrategy):
    def __init__(self, k_list=[1, 3, 5]):
        self.k_list = k_list

    def evaluate(self, output: torch.Tensor, target: torch.Tensor) -> dict:
        probs = torch.softmax(output, dim=1)
        results = {}

        for k in self.k_list:
            _, top_k_indices = probs.topk(k, dim=1)
            correct_k = top_k_indices.eq(target.view(-1, 1).expand_as(top_k_indices))
            acc = correct_k.any(dim=1).float().mean().item()
            results[f"top_{k}_acc"] = acc

        return results
