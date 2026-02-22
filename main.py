from args.arg_parser import parse_args
from strategies.model_strategy.context import ModelContext
from factories import create_model_strategy
import os
import pkgutil
import importlib


import strategies.model_strategy.strategys as model_mods
import strategies.evaluation_strategy.strategies as eval_mods


def load_plugins():
    for m in [model_mods, eval_mods]:
        for loader, name, is_pkg in pkgutil.walk_packages(m.__path__, m.__name__ + "."):
            importlib.import_module(name)


def main():
    args = parse_args()
    load_plugins()

    config = {
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    strategy = create_model_strategy(config)
    ctx = ModelContext(strategy)
    ctx.prepare()
    ctx.build()
    ctx.train()

    results = ctx.evaluate()
    print(
        f"Completed minimal run for experiment '{args.experiment_name}'; results={results}"
    )


if __name__ == "__main__":
    main()
