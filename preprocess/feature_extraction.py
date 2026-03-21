"""
src/data/preprocessing.py
音声セグメントからの特徴量抽出。
"""

import numpy as np
import librosa
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """音声特徴量抽出の設定。"""

    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    fmin: float = 20.0
    fmax: float = 8000.0
    mfcc_n_coeffs: int = 13

    @classmethod
    def from_dict(cls, d: dict) -> "AudioConfig":
        return cls(
            n_mels=d.get("n_mels", 128),
            n_fft=d.get("n_fft", 1024),
            hop_length=d.get("hop_length", 512),
            fmin=d.get("fmin", 20.0),
            fmax=d.get("fmax", 8000.0),
            mfcc_n_coeffs=d.get("mfcc_n_coeffs", 13),
        )


def compute_mel_spectrogram(
    audio: np.ndarray, sr: int, config: AudioConfig
) -> np.ndarray:
    """
    メルスペクトログラムを計算する。

    Parameters
    ----------
    audio : np.ndarray
        音声波形。形状: (num_samples,)。dtype: float32
    sr : int
        サンプリングレート（Hz）
    config : AudioConfig
        音声設定

    Returns
    -------
    mel_spec_db : np.ndarray
        デシベルスケールのメルスペクトログラム。
        形状: (n_mels, time_steps)
        time_steps = floor(num_samples / hop_length) + 1
        値域: [-80.0, 0.0]（ref=np.maxの場合）

    Notes
    -----
    - fmin=20Hz, fmax=8000Hz はブレス音の主要帯域に対応
    - パワースペクトログラム→デシベル変換を適用
    - 無限大・NaN は発生しない（librosa.power_to_db は内部でepsクリッピング）
    """
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,  # パワースペクトログラム
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)

    # NaN/Inf チェック（防御的プログラミング）
    assert not np.any(np.isnan(mel_spec_db)), "メルスペクトログラムにNaNが含まれる"
    assert not np.any(np.isinf(mel_spec_db)), "メルスペクトログラムにInfが含まれる"

    return mel_spec_db


def compute_mfcc(
    audio: np.ndarray,
    sr: int,
    config: AudioConfig,
    include_delta: bool = True,
    include_delta_delta: bool = True,
) -> np.ndarray:
    """
    MFCC（メル周波数ケプストラム係数）を計算する。

    Parameters
    ----------
    audio : np.ndarray
        音声波形。形状: (num_samples,)
    sr : int
        サンプリングレート（Hz）
    config : AudioConfig
        音声設定
    include_delta : bool
        Δ（一次差分）を含めるか。デフォルト: True
    include_delta_delta : bool
        ΔΔ（二次差分）を含めるか。デフォルト: True

    Returns
    -------
    features : np.ndarray
        MFCC特徴量。
        形状: (n_features, time_steps)
        n_features = n_coeffs * (1 + include_delta + include_delta_delta)
        つまりデフォルトでは 13 * 3 = 39
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=config.mfcc_n_coeffs,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        fmin=config.fmin,
        fmax=config.fmax,
    )

    features_list = [mfcc]

    if include_delta:
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        features_list.append(mfcc_delta)

    if include_delta_delta:
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features_list.append(mfcc_delta2)

    features = np.concatenate(features_list, axis=0)
    return features


def compute_spectral_features(audio: np.ndarray, sr: int, config: AudioConfig) -> dict:
    """
    追加のスペクトル特徴量を計算する。

    Parameters
    ----------
    audio : np.ndarray
        音声波形。形状: (num_samples,)
    sr : int
        サンプリングレート（Hz）
    config : AudioConfig
        音声設定

    Returns
    -------
    features : dict
        キー: 特徴名、値: np.ndarray（形状: (time_steps,)）
        含まれるキー:
        - "spectral_centroid": スペクトル重心（Hz）
        - "spectral_rolloff": スペクトルロールオフ（Hz）
        - "spectral_bandwidth": スペクトル帯域幅（Hz）
        - "zero_crossing_rate": ゼロ交差率
    """
    features = {}

    features["spectral_centroid"] = librosa.feature.spectral_centroid(
        y=audio, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length
    )[0]

    features["spectral_rolloff"] = librosa.feature.spectral_rolloff(
        y=audio, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length
    )[0]

    features["spectral_bandwidth"] = librosa.feature.spectral_bandwidth(
        y=audio, sr=sr, n_fft=config.n_fft, hop_length=config.hop_length
    )[0]

    features["zero_crossing_rate"] = librosa.feature.zero_crossing_rate(
        y=audio, frame_length=config.n_fft, hop_length=config.hop_length
    )[0]

    return features


def normalize_waveform(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    生波形を正規化し、長さを調整する（1D-CNN用）。

    正規化方法: z-score正規化（平均0、標準偏差1）

    Parameters
    ----------
    audio : np.ndarray
        音声波形。形状: (num_samples,)
    target_length : int
        目標サンプル数

    Returns
    -------
    audio_normalized : np.ndarray
        正規化された波形。形状: (target_length,)。dtype: float32

    Notes
    -----
    - 標準偏差が極小（< 1e-8）の場合はゼロベクトルを返す（完全無音）
    - 短い場合: 右側ゼロパディング
    - 長い場合: 先頭から target_length 分を使用
    """
    std = np.std(audio)

    if std < 1e-8:
        # 完全無音：ゼロベクトルを返す
        return np.zeros(target_length, dtype=np.float32)

    audio_normalized = ((audio - np.mean(audio)) / std).astype(np.float32)

    if len(audio_normalized) < target_length:
        audio_normalized = np.pad(
            audio_normalized,
            (0, target_length - len(audio_normalized)),
            mode="constant",
            constant_values=0.0,
        )
    elif len(audio_normalized) > target_length:
        audio_normalized = audio_normalized[:target_length]

    assert len(audio_normalized) == target_length
    return audio_normalized
