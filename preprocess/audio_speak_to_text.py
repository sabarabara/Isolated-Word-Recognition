import os
import re

import whisper
from pydub import AudioSegment

# ==========================================
# 設定エリア
# ==========================================
TARGET_DIR = "outputs/wavs/output_12wavs"  # 分割済みファイルがある場所
LOG_FILE = "outputs/stt/transcription_large12.txt"
MODEL_SIZE = "large-v3"  # 軽量で速いモデル
LANGUAGE = "ja"


def natural_keys(text):
    """
    リストの要素（文字列）を「自然順（1, 2, 10...）」で並べ替えるためのキー
    """
    return [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", text)]


def main():
    print(f"Whisperモデル({MODEL_SIZE})を読み込み中...")
    model = whisper.load_model(MODEL_SIZE)

    # 1. wavファイルを取得
    files = [
        f
        for f in os.listdir(TARGET_DIR)
        if f.endswith(".wav") and not f.startswith("silence")
    ]

    # 2. 【ここが修正ポイント】自然順でソート
    files.sort(key=natural_keys)

    if not files:
        print(f"エラー: {TARGET_DIR} 内に対象ファイルがありません。")
        return

    print(f"対象ファイル順: {files}")  # 確認用に順番を表示

    results = []
    for filename in files:
        file_path = os.path.join(TARGET_DIR, filename)
        audio = AudioSegment.from_file(file_path)
        duration = audio.duration_seconds

        print(f"処理中: {filename} ({duration:.2f}s)... ", end="", flush=True)

        result = model.transcribe(file_path, language=LANGUAGE)
        text = result["text"].strip()

        line = f"[{filename}] ({duration:.2f}s): {text}"
        results.append(line)
        print("完了")

    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(results))

    print(f"\n完了！ 順番通りに保存されました: {LOG_FILE}")


if __name__ == "__main__":
    main()
