from strategies.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from strategies.model_strategy.model_strategy import ModelStrategy
from strategies.evaluation_strategy.composite import CompositeEvaluation
from strategies.registry import MODEL_REGISTRY, EVAL_REGISTRY


def create_eval_strategy(config: dict) -> EvaluationStrategy:
    strategies = [
        EVAL_REGISTRY.get(k)(config)
        for k in ["topk", "confusion"]
        if config.get(f"use_{k}") and EVAL_REGISTRY.get(k)
    ]

    if not strategies:
        mock_cls = EVAL_REGISTRY.get("mock")
        return mock_cls() if mock_cls else None

    return strategies[0] if len(strategies) == 1 else CompositeEvaluation(strategies)


def create_model_strategy(config: dict) -> ModelStrategy:
    model_type = config.get("model_type", "simple")
    eval_strat = create_eval_strategy(config)
    model_cls = MODEL_REGISTRY.get(model_type) or MODEL_REGISTRY.get("mock")

    if not model_cls:
        raise ValueError(f"Model type '{model_type}' and 'mock' are not registered.")

    return model_cls(eval_strategy=eval_strat, config=config)
