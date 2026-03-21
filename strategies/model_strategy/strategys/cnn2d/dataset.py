import json
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from utils.logging import setup_logging
from preprocess.feature_extraction import AudioConfig, compute_mel_spectrogram

logger = setup_logging(__name__)

class CNN2DDataset(Dataset):

    def __init__(
        self,
        annotation_dir : Path,
        audio_dir : Path,
        config: dict,  # のちに変更
        segment_duration : Optional[float] = 0.1,
        augmentation :  Optional[callable] = None,
        quality : Optional[list] = None
    ):
        self.config = config
        self.segment_duration = segment_duration
        self.augmentation = augmentation
        self.quality = quality
        self.samples = self._load_samples(annotation_dir, audio_dir)
        logger.info(f"データセットサイズ: {len(self.samples)} サンプル")

    def _load_samples(self, annotation_dir: Path, audio_dir: Path) -> list:

        samples = []
        annotation_files = sorted([f for f in annotation_dir.glob("*.json") if f.name != "example_annotation.json"])

        if len(annotation_files) == 0:
            raise ValueError(f"アノテーションファイルが見つかりません: {annotation_dir}")
        
        all_card_ids = set()
        for ann_path in annotation_files:
            with open(ann_path, "r", encoding="utf-8") as _f:
                ann_data = json.load(_f)
            for card in ann_data["cards"]:
                all_card_ids.add(card["card_id"])

        card_id_to_label = {card_id: idx for idx, card_id in enumerate(sorted(all_card_ids))}
        logger.info(f"カードIDとラベルのマッピング: {card_id_to_label}")

        for ann_path in annotation_files:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            session_id = data["session_id"]
            reader_id = data.get("reader_id", "unknown_reader")

            for card in data["cards"]:

                silence_file = card.get("silence_file")
                if silence_file is None:
                    logger.warning(f"カード{card['card_id']}のsilence_fileが見つかりません。スキップします。")
                    continue
                audio_path = audio_dir / silence_file

                if not audio_path.exists():
                    logger.warning(f"カード{card['card_id']}の音声ファイルが見つかりません: {audio_path}")
                    continue
                
                card_id = card["card_id"]
                card_label = card_id_to_label[card_id]

                samples.append({
                    "audio_path": str(audio_path),
                    "session_id": session_id,
                    "reader_id": reader_id,
                    "card_id": card_id,
                    "card_label": card_label,  # 0-99の範囲に変換されたラベル
                    "card_text": card.get("card_text", ""),
                    # メタデータ（存在しない場合はデフォルト値）
                    "initial_phoneme": card.get("initial_phoneme", "unknown"),
                    "initial_phoneme_category": card.get("initial_phoneme_category", "unknown"),
                    "articulation_place": card.get("articulation_place", "unknown"),
                    "articulation_manner": card.get("articulation_manner", "unknown"),
                    "kimariji_length": card.get("kimariji_length", 0),
                    "silence_duration_sec": data.get("silence_duration_sec", 0.5),
                })

        return samples
    
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.samples[idx]
        audio, sr = librosa.load(sample["audio_path"], sr=self.config.get("sample_rate", 44100))

        if self.augmentation is not None:
            audio = self.augmentation(audio, sr)

        features = compute_mel_spectrogram(audio, sr, AudioConfig.from_dict(self.config))
        features_tensor = torch.tensor(features,dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(sample["card_label"], dtype=torch.long)

        metadata = {
            "card_id": sample["card_id"],
            "reader_id": sample["reader_id"],
            "session_id": sample["session_id"],
            "card_text": sample["card_text"],
            "initial_phoneme": sample["initial_phoneme"],
            "initial_phoneme_category": sample["initial_phoneme_category"],
            "articulation_place": sample["articulation_place"],
            "articulation_manner": sample["articulation_manner"],
            "kimariji_length": sample["kimariji_length"],
            "silence_duration_sec": sample["silence_duration_sec"],
        }

        return features_tensor, label, metadata
