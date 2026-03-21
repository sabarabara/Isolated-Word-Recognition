"""CNN1D 用データセット。波形を正規化しで1D入力として返す。"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from strategies.model_strategy.strategys._shared.base_dataset import BaseDataset
from preprocess.feature_extraction import normalize_waveform


class CNN1DDataset(BaseDataset):
    """(waveform_tensor, label, metadata) 形式で返すデータセット。

    waveform_tensor shape: (1, target_length)
    """

    def __init__(
        self,
        annotation_dir: Path,
        audio_dir: Path,
        config: dict,
        augmentation: Optional[callable] = None,
    ):
        super().__init__(annotation_dir, audio_dir, config, augmentation)
        sr = config.get("sample_rate", 44100)
        duration = config.get("waveform_duration", 0.6)
        self.target_length = int(sr * duration)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        audio, sr = self._load_audio(sample["audio_path"])
        waveform = normalize_waveform(audio, self.target_length)  # (target_length,)
        features_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # (1, T)
        label = torch.tensor(sample["card_label"], dtype=torch.long)
        return features_tensor, label, self._make_metadata(sample)
