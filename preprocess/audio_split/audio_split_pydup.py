import os

from pydub import AudioSegment

# ==========================================
# 設定エリア
# ==========================================
INPUT_FILE = "data/audio/raw/singer2/5_+100.wav"
OUTPUT_DIR = "outputs/wavs/output_12wavs"
LOG_FILE = "outputs/texts/output_12wavs.txt"

# 判定基準
SILENCE_THRESH = -20.0  # 区間を検出するための閾値
MIN_SILENCE_LEN = 1000
MIN_SOUND_LEN = 3000

# 【重要】本編の開始を確定させる閾値
# ノーマライズ後、この音量を超えるまでは「まだ無音（ブレス）」として扱います
VOCAL_START_THRESH = -12.0

CHUNK_SIZE = 10


def get_peak_dBFS(chunk):
    if len(chunk) == 0:
        return -float("inf")
    return chunk.max_dBFS


def main():
    print(f"音声を読み込み中...: {INPUT_FILE}")
    try:
        audio = AudioSegment.from_file(INPUT_FILE)
    except FileNotFoundError:
        print("エラー: ファイルが見つかりません。")
        return

    # 1. ノーマライズで基準を揃える
    print("音量を正規化中...")
    audio = audio.normalize()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. 一次解析（広めに区間を特定）
    print("解析中...")
    nonsilent_ranges = []
    is_in_sound = False
    start_ms = 0

    for i in range(0, len(audio), CHUNK_SIZE):
        chunk = audio[i : i + CHUNK_SIZE]
        peak = get_peak_dBFS(chunk)
        if peak >= SILENCE_THRESH:
            if not is_in_sound:
                start_ms = i
                is_in_sound = True
        else:
            if is_in_sound:
                lookahead_end = i + MIN_SILENCE_LEN
                if lookahead_end > len(audio):
                    lookahead_end = len(audio)
                lookahead_chunk = audio[i:lookahead_end]
                if get_peak_dBFS(lookahead_chunk) < SILENCE_THRESH:
                    end_ms = i
                    nonsilent_ranges.append([start_ms, end_ms])
                    is_in_sound = False

    if is_in_sound:
        nonsilent_ranges.append([start_ms, len(audio)])

    valid_ranges = [r for r in nonsilent_ranges if (r[1] - r[0]) >= MIN_SOUND_LEN]

    if len(valid_ranges) < 2:
        print("エラー: 有効な区間が見つかりませんでした。")
        return

    # 3. 書き出し処理（ブレスを無音側へ移動）
    print("\n--- 書き出しを開始します（ブレスを無音側に回します） ---")
    log_lines = []

    for i in range(1, len(valid_ranges)):
        prev_end = valid_ranges[i - 1][1]
        curr_start = valid_ranges[i][0]
        curr_end = valid_ranges[i][1]

        # 本編区間の中から、本当の声の始まり（VOCAL_START_THRESH超え）を探す
        temp_chunk = audio[curr_start:curr_end]
        vocal_offset = 0
        for ms in range(len(temp_chunk)):
            if temp_chunk[ms : ms + 1].max_dBFS > VOCAL_START_THRESH:
                vocal_offset = ms
                break

        # 修正された境界点
        # ブレス部分はここまで（無音ファイルの終点）
        final_silence_end = curr_start + vocal_offset
        # 声はここから（本編ファイルの始点）
        final_sound_start = final_silence_end

        # [無音ファイル] 前の終わり 〜 厳密な声の開始まで（ここにブレスが入る）
        silence_chunk = audio[prev_end:final_silence_end]
        silence_name = f"silence {i:02d}.wav"
        silence_chunk.export(os.path.join(OUTPUT_DIR, silence_name), format="wav")

        # [本編ファイル] 厳密な声の開始 〜 終わり
        sound_chunk = audio[final_sound_start:curr_end]
        sound_name = f"{i:02d}.wav"
        sound_duration = sound_chunk.duration_seconds
        sound_chunk.export(os.path.join(OUTPUT_DIR, sound_name), format="wav")

        line = f"{sound_name}: {sound_duration:.2f}s (ブレス {vocal_offset}ms を無音側に移動)"
        print(line)
        log_lines.append(line)

    if not os.path.exists(os.path.dirname(LOG_FILE)):
        os.makedirs(os.path.dirname(LOG_FILE))
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\nすべて完了！")


if __name__ == "__main__":
    main()
