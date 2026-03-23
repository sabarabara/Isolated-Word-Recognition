"""Common lazy-import bundle for model strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class StrategyLazyImports:
    """Container returned by each strategy's _lazy_imports().

    It remains backward-compatible with existing tuple unpacking.
    """

    dataset_cls: type
    model_cls: type
    trainer_cls: type
    evaluator_cls: type
    default_cam_layer: str | None = None

    def __iter__(self) -> Iterator[type]:
        # Keep compatibility with: dataset_cls, model_cls, trainer_cls, evaluator_cls = _lazy_imports()
        yield self.dataset_cls
        yield self.model_cls
        yield self.trainer_cls
        yield self.evaluator_cls
