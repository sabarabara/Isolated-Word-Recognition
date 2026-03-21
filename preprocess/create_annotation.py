#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Dict

# ==========================================
# ### 設定エリア：ここを書き換えてください ###
# ==========================================
# 1ファイルだけ処理したいなら 'single'、フォルダごとやりたいなら 'batch'
MODE = "batch"

# [Single用設定]
SINGLE_TEXT = Path("../data/text/01.txt")
SINGLE_WAVS = Path("../data/outputs/segment/0.5s/01wavs")
SINGLE_OUT = Path("../data/annotations/01.json")

# [Batch用設定]
BATCH_TEXT_DIR = Path("../data/text")
BATCH_WAVS_BASE = Path("../data/outputs/segment/0.5s/")
BATCH_OUT_DIR = Path("../data/annotation_data")

READER_ID = "reader01"
# ==========================================


def parse_text_file(text_file: Path) -> Dict[int, str]:
    card_texts = {}
    if not text_file.exists():
        return card_texts
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"\[(\d+)\.wav\]\s*(.+)", line)
            if match:
                card_texts[int(match.group(1))] = match.group(2).strip()
    return card_texts


def create_simple_annotation(
    text_file: Path, wavs_dir: Path, output_file: Path, session_id: str
):
    card_texts = parse_text_file(text_file)
    if not card_texts:
        print(f"× スキップ (テキスト空): {text_file}")
        return

    annotation = {
        "session_id": session_id,
        "reader_id": READER_ID,
        "sampling_rate": 44100,
        "silence_duration_sec": 0.5,
        "cards": [],
    }

    for card_num, card_text in sorted(card_texts.items()):
        if card_num == 1:
            continue  # 序歌スキップ

        audio_file = wavs_dir / f"{card_num:02d}.wav"
        if not audio_file.exists():
            continue

        # 相対パス計算（実行環境に合わせて調整してください）
        # ここでは wavs_dir の2階層上からの相対パスにしています
        try:
            rel_path = str(audio_file.relative_to(wavs_dir.parent.parent))
        except ValueError:
            rel_path = str(audio_file)

        annotation["cards"].append(
            {
                "card_id": card_num,
                "card_label": card_num,
                "card_text": card_text,
                "silence_file": rel_path,
                "utterance_file": None,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)
    print(f"✓ 作成完了: {output_file} ({len(annotation['cards'])}枚)")


def main():
    if MODE == "single":
        create_simple_annotation(SINGLE_TEXT, SINGLE_WAVS, SINGLE_OUT, SINGLE_TEXT.stem)

    else:  # batch
        text_files = sorted(BATCH_TEXT_DIR.glob("*.txt"))
        for tf in text_files:
            wav_dir = BATCH_WAVS_BASE / f"{tf.stem}wavs"
            if wav_dir.exists():
                create_simple_annotation(
                    tf, wav_dir, BATCH_OUT_DIR / f"{tf.stem}.json", tf.stem
                )
            else:
                print(f"! 警告: ディレクトリが見つかりません: {wav_dir}")


if __name__ == "__main__":
    main()
