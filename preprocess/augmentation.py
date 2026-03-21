"""音声データ拡張ユーティリティ。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AugmentationConfig:
    noise_probability: float = 0.5
    noise_min_snr_db: float = 10.0
    noise_max_snr_db: float = 25.0
    gain_probability: float = 0.3
    gain_min_db: float = -3.0
    gain_max_db: float = 3.0


class AudioAugmentation:
    def __init__(self, config: AugmentationConfig, random_seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(random_seed)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        データ拡張を適用する。

        Parameters
        ----------
        audio : np.ndarray
            音声波形。形状: (num_samples,)
        sr : int
            サンプリングレート

        Returns
        -------
        augmented : np.ndarray
            拡張後の波形。形状は入力と同一。
        """
        augmented = audio.copy()

        # ガウシアンノイズ
        if self.rng.random() < self.config.noise_probability:
            augmented = self._add_noise(augmented)

        # ゲイン調整
        if self.rng.random() < self.config.gain_probability:
            augmented = self._adjust_gain(augmented)

        return np.clip(augmented, -1.0, 1.0).astype(audio.dtype)

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """SNRに基づくガウシアンノイズの付加。"""
        snr_db = self.rng.uniform(
            self.config.noise_min_snr_db, self.config.noise_max_snr_db
        )
        signal_power = np.mean(audio**2)

        if signal_power < 1e-10:
            return audio  # 無音にノイズを足しても意味がない

        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.normal(0, np.sqrt(noise_power), size=len(audio))
        return (audio + noise).astype(audio.dtype)

    def _adjust_gain(self, audio: np.ndarray) -> np.ndarray:
        """ゲイン（音量）の調整。"""
        gain_db = self.rng.uniform(self.config.gain_min_db, self.config.gain_max_db)
        gain_linear = 10 ** (gain_db / 20)
        return (audio * gain_linear).astype(audio.dtype)


def build_audio_augmentation(config: dict):
    """config(dict) から AudioAugmentation を構築する。"""
    if not config.get("enable_augmentation", False):
        return None

    aug_cfg = AugmentationConfig(
        noise_probability=float(config.get("noise_probability", 0.5)),
        noise_min_snr_db=float(config.get("noise_min_snr_db", 10.0)),
        noise_max_snr_db=float(config.get("noise_max_snr_db", 25.0)),
        gain_probability=float(config.get("gain_probability", 0.3)),
        gain_min_db=float(config.get("gain_min_db", -3.0)),
        gain_max_db=float(config.get("gain_max_db", 3.0)),
    )
    return AudioAugmentation(aug_cfg, random_seed=int(config.get("seed", 42)))
