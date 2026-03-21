from pathlib import Path
import os
import pkgutil
import importlib

import hydra
from omegaconf import DictConfig

from strategies.model_strategy.context import ModelContext
from factories import create_model_strategy


import strategies.model_strategy.strategys as model_mods
import strategies.evaluation_strategy.strategies as eval_mods


def load_plugins():
    for m in [model_mods, eval_mods]:
        for loader, name, is_pkg in pkgutil.walk_packages(m.__path__, m.__name__ + "."):
            try:
                importlib.import_module(name)
            except Exception as e:
                print(f"[load_plugins] skipped '{name}': {e}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    load_plugins()

    output_dir = str(Path(cfg.output_dir) / cfg.model_type)

    config = {
        "model_type": cfg.model_type,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "annotation_dir": cfg.annotation_dir,
        "audio_dir": str(Path(cfg.data_dir).parent / "outputs" / "segment"),
        "output_dir": output_dir,
        "segment_duration": cfg.segment_duration,
        "input_type": cfg.input_type,
        "seed": cfg.seed,
        "use_topk": cfg.use_topk,
        "use_confusion": cfg.use_confusion,
    }

    os.makedirs(output_dir, exist_ok=True)

    strategy = create_model_strategy(config)
    ctx = ModelContext(strategy)
    ctx.prepare()
    ctx.build()
    ctx.train()

    results = ctx.evaluate()
    print(
        f"Completed minimal run for experiment '{cfg.experiment_name}'; results={results}"
    )


if __name__ == "__main__":
    main()
