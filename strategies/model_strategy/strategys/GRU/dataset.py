"""GRU 用データセット。メルスペクトログラムを時系列 (T, n_mels) で返す。"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from strategies.model_strategy.strategys._shared.base_dataset import KarutaBaseDataset
from preprocess.feature_extraction import AudioConfig, compute_mel_spectrogram


class GRUDataset(KarutaBaseDataset):
    """(mel_seq_tensor, label, metadata) 形式で返すデータセット。

    mel_seq_tensor shape: (T, n_mels)  ← RNN の batch_first=True 入力に対応
    """

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio, sr = self._load_audio(sample["audio_path"])

        mel = compute_mel_spectrogram(audio, sr, AudioConfig.from_dict(self.config))
        # mel: (n_mels, T) → transpose to (T, n_mels)
        features_tensor = torch.tensor(mel.T, dtype=torch.float32)

        label = torch.tensor(sample["card_label"], dtype=torch.long)
        return features_tensor, label, self._make_metadata(sample)
