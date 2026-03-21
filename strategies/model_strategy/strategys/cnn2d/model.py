import torch
import torch.nn as nn
from configs.experiment_config import ModelConfig


class CNN2DModel(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1: 微細なブレスパターンの検出
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=config.dropout_conv),

            # Block 2: 周波数帯域の組み合わせパターン
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=config.dropout_conv),

            # Block 3: 高次の音響特徴の統合
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(config.adaptive_pool_output_2d),
            nn.Dropout2d(p=config.dropout_final_conv),
        )

        pool_h, pool_w = config.adaptive_pool_output_2d
        flatten_dim = 128 * pool_h * pool_w

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, config.fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_fc),
            nn.Linear(config.fc_hidden_dim, config.num_classes),
        )

        # 重み初期化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv_layers(x)
        logits = self.classifier(x)
        return logits
