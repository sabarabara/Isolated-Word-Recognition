import torch
from pathlib import Path
from typing import Optional

from strategies.model_strategy.strategys._shared.base_dataset import BaseDataset
from preprocess.feature_extraction import AudioConfig, compute_mel_spectrogram


class CNN2DDataset(BaseDataset):
    def __init__(
        self,
        annotation_dir: Path,
        audio_dir: Path,
        config: dict,  # のちに変更
        segment_duration: Optional[float] = 0.1,
        augmentation: Optional[callable] = None,
        quality: Optional[list] = None,
    ):
        self.segment_duration = segment_duration
        self.quality = quality
        super().__init__(annotation_dir, audio_dir, config, augmentation)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]
        audio, sr = self._load_audio(sample["audio_path"])

        features = compute_mel_spectrogram(
            audio, sr, AudioConfig.from_dict(self.config)
        )
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(sample["card_label"], dtype=torch.long)
        return features_tensor, label, self._make_metadata(sample)
