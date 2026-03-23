"""Lightweight Conformer-like model (CNN frontend + Transformer encoder)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFrontend(nn.Module):
    """CNN frontend that converts (B, 1, n_mels, T) -> (B, T', d_model)."""

    def __init__(
        self,
        d_model: int,
        dropout_conv: float,
        dropout_final_conv: float,
        adaptive_pool_output_2d: tuple[int, int],
    ):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 微細なブレスパターンの検出
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),
            # Block 2: 周波数帯域の組み合わせパターン
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_conv),
            # Block 3: 高次の音響特徴の統合
            nn.Conv2d(
                in_channels=64,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(adaptive_pool_output_2d),
            nn.Dropout2d(p=dropout_final_conv),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)  # (B, D, F', T')
        x = x.mean(dim=2)  # (B, D, T')
        return x.transpose(1, 2)  # (B, T', D)


class TransformerBackbone(nn.Module):
    """Transformer encoder over temporal sequence (B, T, D)."""

    def __init__(self, d_model: int, nhead: int, num_layers: int, dropout: float):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.norm(x)


class ClassificationHead(nn.Module):
    """Pooling + MLP classifier."""

    def __init__(self, d_model: int, num_classes: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)  # temporal global average pooling
        return self.head(x)


class ConformerModel(nn.Module):
    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}

        self.n_mels = int(cfg.get("n_mels", 128))
        self.max_frames = int(cfg.get("max_frames", 128))
        num_classes = int(cfg.get("num_classes", 100))

        d_model = int(cfg.get("conformer_d_model", 192))
        nhead = int(cfg.get("conformer_nhead", 4))
        num_layers = int(cfg.get("conformer_num_layers", 4))
        dropout = float(cfg.get("conformer_dropout", 0.1))
        dropout_conv = float(cfg.get("conformer_dropout_conv", 0.2))
        dropout_final_conv = float(cfg.get("conformer_dropout_final_conv", 0.3))
        adaptive_pool_output_2d = tuple(cfg.get("conformer_adaptive_pool_output_2d", [4, 4]))

        # Split modules so each part can be swapped/customized independently.
        self.cnn_frontend = CNNFrontend(
            d_model=d_model,
            dropout_conv=dropout_conv,
            dropout_final_conv=dropout_final_conv,
            adaptive_pool_output_2d=adaptive_pool_output_2d,
        )
        self.transformer_backbone = TransformerBackbone(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T) -> fixed size for stable training.
        x = F.adaptive_avg_pool2d(x, (self.n_mels, self.max_frames))
        return self.cnn_frontend(x)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_backbone(x)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.encode_sequence(x)
        return self.classify(x)
