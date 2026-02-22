import torch.nn as nn


class CNN1DModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=100, base_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(base_channels * 2, num_classes)

    def forward(self, x):
        # x: (batch, channels, length)
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)
