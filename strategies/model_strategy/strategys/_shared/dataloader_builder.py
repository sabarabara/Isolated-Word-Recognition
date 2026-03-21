"""Shared DataLoader construction helpers for model strategies."""

from torch.utils.data import DataLoader, Subset
import torch


def build_train_val_loaders(
    dataset_cls,
    annotation_dir,
    audio_dir,
    config: dict,
    batch_size: int,
    train_augmentation=None,
):
    """Create train/val loaders with identical split indices.

    Train and validation use separate dataset instances so augmentation can be
    enabled only for training.
    """
    base_dataset = dataset_cls(
        annotation_dir,
        audio_dir,
        config=config,
        augmentation=None,
    )
    n_total = len(base_dataset)
    n_val = max(1, int(0.2 * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(int(config.get("seed", 42)))
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset_full = dataset_cls(
        annotation_dir,
        audio_dir,
        config=config,
        augmentation=train_augmentation,
    )
    val_dataset_full = dataset_cls(
        annotation_dir,
        audio_dir,
        config=config,
        augmentation=None,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, n_train, n_val
