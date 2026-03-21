"""AST 用データセット。CNN2DDataset と同形式 (1, n_mels, T)。"""
import torch
from pathlib import Path
from typing import Optional

from strategies.model_strategy.strategys._shared.base_dataset import BaseDataset
from preprocess.feature_extraction import AudioConfig, compute_mel_spectrogram


class ASTDataset(BaseDataset):
    """(mel_2d_tensor, label, metadata) 形式で返すデータセット。

    mel_2d_tensor shape: (1, n_mels, T)
    """

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio, sr = self._load_audio(sample["audio_path"])

        mel = compute_mel_spectrogram(audio, sr, AudioConfig.from_dict(self.config))
        # mel: (n_mels, T) → add channel dim → (1, n_mels, T)
        features_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(sample["card_label"], dtype=torch.long)
        return features_tensor, label, self._make_metadata(sample)
