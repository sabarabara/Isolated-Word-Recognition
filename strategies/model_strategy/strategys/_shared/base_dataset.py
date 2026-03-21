"""全モデル共通の Karuta データセット基底クラス。

サブクラスは __getitem__ をオーバーライドして各モデル用の特徴量テンソルを返す。
戻り値は (features_tensor, label, metadata) の 3-tuple に統一する。
"""
import json
import librosa
import torch
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from utils.logging import setup_logging

logger = setup_logging(__name__)


class BaseDataset(Dataset):

    def __init__(
        self,
        annotation_dir: Path,
        audio_dir: Path,
        config: dict,
        augmentation: Optional[callable] = None,
    ):
        self.config = config
        self.augmentation = augmentation
        self.samples = self._load_samples(Path(annotation_dir), Path(audio_dir))
        logger.info(f"データセットサイズ: {len(self.samples)} サンプル")

    def _load_samples(self, annotation_dir: Path, audio_dir: Path) -> list:
        samples = []
        annotation_files = sorted([
            f for f in annotation_dir.glob("*.json")
            if f.name != "example_annotation.json"
        ])
        if not annotation_files:
            raise ValueError(f"アノテーションファイルが見つかりません: {annotation_dir}")

        all_card_ids = set()
        for ann_path in annotation_files:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann_data = json.load(f)
            for card in ann_data["cards"]:
                all_card_ids.add(card["card_id"])

        card_id_to_label = {cid: idx for idx, cid in enumerate(sorted(all_card_ids))}

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
                    logger.warning(f"音声ファイルが見つかりません: {audio_path}")
                    continue

                card_id = card["card_id"]
                samples.append({
                    "audio_path": str(audio_path),
                    "session_id": session_id,
                    "reader_id": reader_id,
                    "card_id": card_id,
                    "card_label": card_id_to_label[card_id],
                    "card_text": card.get("card_text", ""),
                    "initial_phoneme": card.get("initial_phoneme", "unknown"),
                    "initial_phoneme_category": card.get("initial_phoneme_category", "unknown"),
                    "articulation_place": card.get("articulation_place", "unknown"),
                    "articulation_manner": card.get("articulation_manner", "unknown"),
                    "kimariji_length": card.get("kimariji_length", 0),
                    "silence_duration_sec": data.get("silence_duration_sec", 0.5),
                })

        return samples

    def _load_audio(self, audio_path: str):
        """音声ファイルを読み込み (audio, sr) を返す。augmentation も適用。"""
        sr = self.config.get("sample_rate", 44100)
        audio, sr_out = librosa.load(audio_path, sr=sr)
        if self.augmentation is not None:
            audio = self.augmentation(audio, sr_out)
        return audio, sr_out

    def _make_metadata(self, sample: dict) -> dict:
        return {
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raise NotImplementedError
