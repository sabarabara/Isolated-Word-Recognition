import random
import torch
from torch.utils.data import Dataset


class RandomAudioDataset(Dataset):
    """Simple dataset that yields random waveform tensors and labels.

    This is intended as a minimal runnable dataset for development and testing.
    """

    def __init__(self, num_samples=256, length=16000, num_classes=100):
        self.num_samples = num_samples
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # waveform shaped (channels=1, length)
        waveform = torch.randn(1, self.length, dtype=torch.float32)
        label = random.randint(0, self.num_classes - 1)
        return waveform, label
