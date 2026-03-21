"""早期停止コールバック。"""
from __future__ import annotations


class EarlyStopping:
    """val metric が改善しなくなったら停止するコールバック。"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best: float | None = None

    def __call__(self, metric: float) -> bool:
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
