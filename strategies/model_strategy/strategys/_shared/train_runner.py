"""Shared training runner for model strategies."""

from pathlib import Path


def run_training(strategy, trainer_cls):
    """Run shared trainer setup and execution for a strategy instance."""
    if strategy.model is None or strategy.train_loader is None:
        print("train: skipped (model or dataloader missing)")
        return None

    trainer = trainer_cls(
        model=strategy.model,
        config=strategy._make_exp_config(),
        train_loader=strategy.train_loader,
        val_loader=strategy.val_loader,
        device=strategy.device,
        output_dir=Path(strategy.config.get("output_dir", ".")),
    )
    strategy.trainer = trainer
    best_metrics = trainer.train()
    print("train: completed")
    return best_metrics
