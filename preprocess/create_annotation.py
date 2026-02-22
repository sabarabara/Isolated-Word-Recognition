#!/usr/bin/env python3
"""
シンプルなアノテーションJSON作成スクリプト
0.5秒のsilence音声から句を予測する学習用
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict


def parse_text_file(text_file: Path) -> Dict[int, str]:
    """
    テキストファイルから句の情報を抽出

    フォーマット例:
    [01.wav] 難波津に 咲くやこの花

    Returns:
        {1: "難波津に 咲くやこの花", ...}
    """
    card_texts = {}

    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # [XX.wav] 句のテキスト のパターンを抽出
            match = re.match(r"\[(\d+)\.wav\]\s*(.+)", line)
            if match:
                card_num = int(match.group(1))
                card_text = match.group(2).strip()
                card_texts[card_num] = card_text

    return card_texts


def create_simple_annotation(
    text_file: Path,
    wavs_dir: Path,
    output_file: Path,
    session_id: str = None,
    reader_id: str = "reader01",
    sampling_rate: int = 44100,
):
    """
    シンプルなアノテーションJSONを作成
    """
    # テキストファイルから句情報を読み込み
    card_texts = parse_text_file(text_file)

    if not card_texts:
        print(f"エラー: {text_file} から句情報を読み取れませんでした")
        return

    # session_idを自動生成
    if session_id is None:
        session_id = wavs_dir.name.replace("output_", "").replace("wavs", "")

    # テキストファイルに記載されているカード番号のみ処理
    print(f"テキストファイルから読み込んだカード: {len(card_texts)} 枚")

    # アノテーション構造を作成
    annotation = {
        "session_id": session_id,
        "reader_id": reader_id,
        "sampling_rate": sampling_rate,
        "silence_duration_sec": 0.5,
        "cards": [],
    }

    # テキストファイルに記載されているカード番号のsilenceファイルのみ処理
    processed_count = 0
    missing_count = 0

    for card_num, card_text in sorted(card_texts.items()):
        # 序歌（card_id=1）をスキップ
        if card_num == 1:
            print("  序歌（card_id=1）をスキップ")
            continue

        # 通常の音声ファイルを使用（2桁ゼロパディング）
        audio_file = wavs_dir / f"{card_num:02d}.wav"

        if not audio_file.exists():
            print(f"警告: カード{card_num}の音声ファイルが見つかりません: {audio_file}")
            missing_count += 1
            continue

        card_data = {
            "card_id": card_num,
            "card_label": card_num,  # そのまま使用（dataset.pyでマッピング）
            "card_text": card_text,
            "silence_file": str(audio_file.relative_to(wavs_dir.parent.parent)),
            "utterance_file": None,
        }

        annotation["cards"].append(card_data)
        processed_count += 1

    # card_idでソート
    annotation["cards"].sort(key=lambda x: x["card_id"])

    # JSON保存
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotation, f, indent=2, ensure_ascii=False)

    print(f"\n✓ アノテーション作成完了: {output_file}")
    print(f"  - カード数: {len(annotation['cards'])} 枚")
    print(f"  - 処理成功: {processed_count} 枚")
    if missing_count > 0:
        print(f"  - 警告: silenceファイル未検出 {missing_count} 枚")
    print(f"  - セッションID: {session_id}")


def batch_create_annotations(
    text_dir: Path, wavs_base_dir: Path, output_dir: Path, reader_id: str = "reader01"
):
    """
    複数のテキストファイルとwavsディレクトリから一括作成
    """
    # テキストファイルを検索
    text_files = sorted(text_dir.glob("*.txt"))

    if not text_files:
        print(f"エラー: {text_dir} に .txt ファイルが見つかりません")
        return

    print(f"{len(text_files)} 個のテキストファイルが見つかりました\n")

    for text_file in text_files:
        # 対応するwavsディレクトリを探す
        # 例: 01.txt -> 01wavs (output_ 接頭辞なし)
        base_name = text_file.stem  # "01"
        wavs_dir = wavs_base_dir / f"{base_name}wavs"

        if not wavs_dir.exists():
            print(f"警告: {wavs_dir} が見つかりません。スキップします。")
            continue

        print(f"{'=' * 60}")
        print(f"処理中: {text_file.name} -> {wavs_dir.name}")
        print("=" * 60)

        # アノテーション作成
        output_file = output_dir / f"{base_name}.json"
        create_simple_annotation(
            text_file=text_file,
            wavs_dir=wavs_dir,
            output_file=output_file,
            session_id=base_name,
            reader_id=reader_id,
        )
        print()

    print(f"{'=' * 60}")
    print(f"✓ 一括作成完了: {len(text_files)} ファイル")
    print(f"出力先: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="0.5秒silence学習用のシンプルなアノテーションJSON作成"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        required=True,
        help="single: 1ファイル作成, batch: 複数ファイル一括作成",
    )
    parser.add_argument(
        "--text-file", type=Path, help="句のテキストファイル (mode=single時)"
    )
    parser.add_argument(
        "--wavs-dir",
        type=Path,
        help="silenceファイルがあるディレクトリ (mode=single時)",
    )
    parser.add_argument(
        "--text-dir", type=Path, help="テキストファイルのディレクトリ (mode=batch時)"
    )
    parser.add_argument(
        "--wavs-base-dir",
        type=Path,
        help="wavsディレクトリの親ディレクトリ (mode=batch時)",
    )
    parser.add_argument(
        "--output", type=Path, help="出力JSONファイルパス (mode=single時)"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="出力ディレクトリ (mode=batch時)"
    )
    parser.add_argument(
        "--session-id", type=str, help="セッションID (省略時は自動生成)"
    )
    parser.add_argument(
        "--reader-id", type=str, default="reader01", help="読み手ID [default: reader01]"
    )

    args = parser.parse_args()

    if args.mode == "single":
        if not args.text_file or not args.wavs_dir or not args.output:
            parser.error(
                "--mode single には --text-file, --wavs-dir, --output が必要です"
            )

        create_simple_annotation(
            text_file=args.text_file,
            wavs_dir=args.wavs_dir,
            output_file=args.output,
            session_id=args.session_id,
            reader_id=args.reader_id,
        )

    elif args.mode == "batch":
        if not args.text_dir or not args.wavs_base_dir or not args.output_dir:
            parser.error(
                "--mode batch には --text-dir, --wavs-base-dir, --output-dir が必要です"
            )

        batch_create_annotations(
            text_dir=args.text_dir,
            wavs_base_dir=args.wavs_base_dir,
            output_dir=args.output_dir,
            reader_id=args.reader_id,
        )


if __name__ == "__main__":
    main()
