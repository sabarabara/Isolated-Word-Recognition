#!/usr/bin/env python3
"""Create offline augmented annotations and WAV files.

For each source annotation JSON, this script writes separate augmented JSON files
such as:
  - 01_aug_1.json
  - 01_aug_2.json
Each output JSON contains augmented cards in the target card-id range.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import librosa
import soundfile as sf

if __package__ is None or __package__ == "":
    # Allow direct execution: python preprocess/create_augmented_dataset.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocess.augmentation import AudioAugmentation, AugmentationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline audio augmentation dataset builder")
    parser.add_argument("--annotation-dir", type=Path, default=Path("data/annotation_data"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/outputs/segment"))
    parser.add_argument("--out-annotation-dir", type=Path, default=Path("data/annotation_data_aug"))
    parser.add_argument("--copies", type=int, default=5, help="Number of augmented JSON files per session")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-card-id", type=int, default=3)
    parser.add_argument("--max-card-id", type=int, default=201)

    parser.add_argument("--noise-probability", type=float, default=0.6)
    parser.add_argument("--noise-min-snr-db", type=float, default=8.0)
    parser.add_argument("--noise-max-snr-db", type=float, default=20.0)
    parser.add_argument("--gain-probability", type=float, default=0.3)
    parser.add_argument("--gain-min-db", type=float, default=-3.0)
    parser.add_argument("--gain-max-db", type=float, default=3.0)
    return parser.parse_args()


def build_augmentor(args: argparse.Namespace) -> AudioAugmentation:
    cfg = AugmentationConfig(
        noise_probability=args.noise_probability,
        noise_min_snr_db=args.noise_min_snr_db,
        noise_max_snr_db=args.noise_max_snr_db,
        gain_probability=args.gain_probability,
        gain_min_db=args.gain_min_db,
        gain_max_db=args.gain_max_db,
    )
    return AudioAugmentation(cfg, random_seed=args.seed)


def augment_session(
    ann_path: Path,
    audio_dir: Path,
    out_annotation_dir: Path,
    copies: int,
    min_card_id: int,
    max_card_id: int,
    augmentor: AudioAugmentation,
) -> tuple[int, int, int]:
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    session_id = str(data.get("session_id", ann_path.stem))
    target_sr = int(data.get("sampling_rate", 44100))
    cards = data.get("cards", [])

    augmented_base_rel = Path("aug") / session_id
    augmented_base_abs = audio_dir / augmented_base_rel
    augmented_base_abs.mkdir(parents=True, exist_ok=True)
    out_annotation_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    for idx, card in enumerate(cards):
        card_id = int(card.get("card_id", idx))
        if min_card_id <= card_id <= max_card_id:
            selected.append((card_id, card))

    total_created = 0
    total_skipped = 0

    for n in range(1, copies + 1):
        out_cards = []
        created_this_copy = 0

        for card_id, card in selected:
            silence_rel = card.get("silence_file")
            if not silence_rel:
                total_skipped += 1
                continue

            src_wav = audio_dir / silence_rel
            if not src_wav.exists():
                total_skipped += 1
                continue

            audio, sr = librosa.load(src_wav.as_posix(), sr=target_sr)
            aug_audio = augmentor(audio, sr)

            aug_filename = f"{card_id:03d}_aug{n:02d}.wav"
            aug_rel = augmented_base_rel / aug_filename
            aug_abs = audio_dir / aug_rel
            sf.write(aug_abs.as_posix(), aug_audio, sr)

            aug_card = dict(card)
            aug_card["silence_file"] = aug_rel.as_posix()
            aug_card["augmented"] = True
            aug_card["augmentation_index"] = n
            out_cards.append(aug_card)

            created_this_copy += 1
            total_created += 1

        out_data = dict(data)
        out_data["cards"] = out_cards
        out_data["augmentation_index"] = n
        out_data["source_annotation"] = ann_path.name

        out_json = out_annotation_dir / f"{session_id}_aug_{n}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        print(
            f"  [aug {n}] {out_json.name}: cards={created_this_copy} (range {min_card_id}-{max_card_id})"
        )

    return total_created, total_skipped, len(selected)


def main() -> None:
    args = parse_args()
    augmentor = build_augmentor(args)

    annotation_files = sorted(args.annotation_dir.glob("*.json"))
    if not annotation_files:
        raise SystemExit(f"No annotation files found in: {args.annotation_dir}")

    total_created = 0
    total_skipped = 0

    for ann_path in annotation_files:
        created, skipped, selected_count = augment_session(
            ann_path=ann_path,
            audio_dir=args.audio_dir,
            out_annotation_dir=args.out_annotation_dir,
            copies=args.copies,
            min_card_id=int(args.min_card_id),
            max_card_id=int(args.max_card_id),
            augmentor=augmentor,
        )
        total_created += created
        total_skipped += skipped
        print(
            f"[ok] {ann_path.name}: selected_cards={selected_count}, created={created}, skipped={skipped}"
        )

    print("---")
    print(f"Finished. augmented_wavs_created={total_created}, skipped_cards={total_skipped}")
    print(f"Output annotation dir: {args.out_annotation_dir}")


if __name__ == "__main__":
    main()
