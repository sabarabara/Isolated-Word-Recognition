"""LSTM 用データセット。GRUDataset と同一形式 (T, n_mels)。"""
import torch
from pathlib import Path
from typing import Optional

from strategies.model_strategy.strategys._shared.base_dataset import BaseDataset
from preprocess.feature_extraction import AudioConfig, compute_mel_spectrogram


class LSTMDataset(BaseDataset):
    """(mel_seq_tensor, label, metadata) 形式で返すデータセット。

    mel_seq_tensor shape: (T, n_mels)
    """

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio, sr = self._load_audio(sample["audio_path"])

        mel = compute_mel_spectrogram(audio, sr, AudioConfig.from_dict(self.config))
        features_tensor = torch.tensor(mel.T, dtype=torch.float32)  # (T, n_mels)

        label = torch.tensor(sample["card_label"], dtype=torch.long)
        return features_tensor, label, self._make_metadata(sample)
