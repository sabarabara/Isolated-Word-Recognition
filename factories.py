from strategies.evaluation_strategy.strategies.mock.simple_eval_strategy import (
    SimpleEvalStrategy,
)
from strategies.model_strategy.strategys.mock.simple_model_strategy import (
    SimpleModelStrategy,
)
from strategies.registry import MODEL_REGISTRY, EVAL_REGISTRY
from strategies.evaluation_strategy.evaluation_strategy import EvaluationStrategy
from strategies.model_strategy.model_strategy import ModelStrategy
from strategies.evaluation_strategy.composite import CompositeEvaluation


def create_eval_strategy(config: dict) -> EvaluationStrategy:
    strategies = []
    for key in ["topk", "confusion"]:
        if config.get(f"use_{key}"):
            cls = EVAL_REGISTRY.get(key)
            if cls:
                strategies.append(cls(config))
    if not strategies:
        return SimpleEvalStrategy()
    if len(strategies) == 1:
        return strategies[0]

    return CompositeEvaluation(strategies)


def create_model_strategy(config: dict) -> ModelStrategy:
    model_type = config.get("model_type", "simple")
    eval_strat = create_eval_strategy(config)
    model_cls = MODEL_REGISTRY.get(model_type) or SimpleModelStrategy

    return model_cls(eval_strategy=eval_strat, config=config)
