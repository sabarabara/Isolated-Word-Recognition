"""簡易 Audio Spectrogram Transformer (AST) モデル。

入力: (batch, 1, n_mels, T)
- AdaptiveAvgPool2d で固定サイズ (n_mels, max_frames) に正規化
- Conv2d パッチ埋め込み
- CLS トークン + Transformer Encoder
- CLS 出力で分類
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASTModel(nn.Module):

    def __init__(self, config: dict = None):
        super().__init__()
        cfg = config or {}
        self.n_mels     = cfg.get("n_mels", 128)
        self.max_frames = cfg.get("max_frames", 64)
        patch_h         = cfg.get("patch_h", 16)
        patch_w         = cfg.get("patch_w", 16)
        d_model         = cfg.get("d_model", 256)
        nhead           = cfg.get("nhead", 4)
        num_layers      = cfg.get("num_transformer_layers", 4)
        dropout         = cfg.get("dropout", 0.1)
        num_classes     = cfg.get("num_classes", 100)

        # パッチ埋め込み (Conv2d stride=patch で非重複分割)
        self.patch_embed = nn.Conv2d(
            1, d_model,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
        )

        n_ph = self.n_mels    // patch_h
        n_pw = self.max_frames // patch_w
        n_patches = n_ph * n_pw

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            dim_feedforward=d_model * 4, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, T) — 可変 T を固定サイズに正規化
        x = F.adaptive_avg_pool2d(x, (self.n_mels, self.max_frames))

        # パッチ埋め込み: (B, d_model, n_ph, n_pw)
        x = self.patch_embed(x)
        B, D, n_ph, n_pw = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)

        # CLS トークン追加
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches+1, d_model)
        x = x + self.pos_embed

        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS トークンのみ
        return self.classifier(x)
