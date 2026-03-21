"""CNN1D モデル。波形 (batch, 1, T) を入力に取る。"""
import torch.nn as nn


class CNN1DModel(nn.Module):
    def __init__(self, config: dict = None):
        cfg = config or {}
        super().__init__()
        in_channels = cfg.get("in_channels", 1)
        base_channels = cfg.get("base_channels", 32)
        num_classes = cfg.get("num_classes", 100)
        dropout = cfg.get("dropout", 0.3)

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, num_classes),
        )

    def forward(self, x):
        # x: (batch, 1, T)
        return self.classifier(self.net(x))
