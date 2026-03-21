"""双方向 GRU モデル。入力 (batch, T, n_mels)。"""
import torch
import torch.nn as nn


class GRUModel(nn.Module):

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}
        input_size  = cfg.get("n_mels", 128)
        hidden_size = cfg.get("hidden_size", 256)
        num_layers  = cfg.get("num_layers", 2)
        dropout     = cfg.get("dropout", 0.3)
        num_classes = cfg.get("num_classes", 100)
        bidirectional = cfg.get("bidirectional", True)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.get("dropout_fc", 0.5)),
            nn.Linear(d, num_classes),
        )
        self.bidirectional = bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, n_mels)
        _, h_n = self.gru(x)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        if self.bidirectional:
            h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h = h_n[-1]
        return self.classifier(h)
