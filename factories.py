def create_eval_strategy(config: dict) -> EvaluationStrategy:
    strategies = []

    if config.get("use_topk"):
        strategies.append(TopKEvaluationStrategy(k_list=config["k_list"]))
    if config.get("use_confusion"):
        strategies.append(ConfusionMatrixStrategy(num_classes=100))

    if len(strategies) > 1:
        return CompositeEvaluation(strategies)
    return strategies[0]


def create_model_strategy(config: dict) -> ModelStrategy:
    eval_strat = create_eval_strategy(config)
    if config["model_type"] == "cnn":
        return HyakuninIsshuCNNStrategy(eval_strategy=eval_strat)
    else:
        raise ValueError("Unknown model type")
