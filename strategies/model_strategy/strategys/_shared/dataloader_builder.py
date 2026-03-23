"""Shared DataLoader construction helpers for model strategies."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split


def build_train_val_loaders(dataset_cls, audio_dir: Path, config: dict, batch_size: int):
    """Create train/val loaders.

    If both train_annotation_dir and val_annotation_dir are provided, use them
    directly (train uses augmented annotations, val uses original annotations).
    Otherwise, split a single annotation_dir with an 80/20 random split.
    """
    train_annotation_dir = config.get("train_annotation_dir")
    val_annotation_dir = config.get("val_annotation_dir")

    if train_annotation_dir and val_annotation_dir:
        train_ds = dataset_cls(Path(train_annotation_dir), audio_dir, config)
        val_ds = dataset_cls(Path(val_annotation_dir), audio_dir, config)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, len(train_ds), len(val_ds)

    annotation_dir = Path(config.get("annotation_dir", "data/annotation_data"))
    ds = dataset_cls(annotation_dir, audio_dir, config)
    n = len(ds)
    n_train = int(n * 0.8)
    n_val = n - n_train
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(config.get("seed", 42))),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_train, n_val
