import os

import librosa
import numpy as np
from pydub import AudioSegment

INPUT_FILE = "data/audio/raw/singer2/5_+100.wav"
OUTPUT_DIR = "outputs/wavs/output_12wavs"
LOG_FILE = "outputs/texts/output_12wavs.txt"

# 判定基準（pydub一次解析用）
SILENCE_THRESH = -20.0
MIN_SILENCE_LEN = 1000
MIN_SOUND_LEN = 3000


VOCAL_START_THRESH_DB = -12.0
DETECTION_MODE = "onset"  # "rms" または "onset" を選択可能


def pydub_to_librosa(audio_segment):
    """pydubのデータをlibrosa形式(numpy)に変換"""
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    # 正規化（16bit想定）
    samples /= 32768.0
    return samples, audio_segment.frame_rate


def detect_vocal_start_librosa(audio_segment, mode="rms"):
    """librosaを使用して精密に声の開始オフセット(ms)を特定"""
    y, sr = pydub_to_librosa(audio_segment)

    if mode == "rms":
        # やり方A: RMS（音量）ベース
        # 1msごとのエネルギーを計算
        hop_length = int(sr / 1000)
        rms = librosa.feature.rms(
            y=y, frame_length=hop_length * 2, hop_length=hop_length
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # しきい値を超える最初のmsを探す
        idx = np.where(rms_db > VOCAL_START_THRESH_DB)[0]
        return int(idx[0]) if len(idx) > 0 else 0

    elif mode == "onset":
        # やり方B: Onset（立ち上がり）ベース
        # 音量変化の「鋭さ」で判定するため、ブレスとの切り分けに強い
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="ms")
        return int(onsets[0]) if len(onsets) > 0 else 0


def main():
    print(f"音声を読み込み中...: {INPUT_FILE}")
    try:
        audio = AudioSegment.from_file(INPUT_FILE)
    except Exception as e:
        print(f"エラー: {e}")
        return

    print("音量を正規化中...")
    audio = audio.normalize()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. pydubによる一次解析（区間の特定）
    print(f"一次解析中 (Mode: {DETECTION_MODE})...")
    nonsilent_ranges = []
    is_in_sound = False
    start_ms = 0
    chunk_size = 10

    # ここはpydubの高速なdBFS判定でざっくり分ける
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i : i + chunk_size]
        if chunk.max_dBFS >= SILENCE_THRESH:
            if not is_in_sound:
                start_ms = i
                is_in_sound = True
        else:
            if is_in_sound:
                lookahead = audio[i : i + MIN_SILENCE_LEN]
                if lookahead.max_dBFS < SILENCE_THRESH:
                    nonsilent_ranges.append([start_ms, i])
                    is_in_sound = False

    if is_in_sound:
        nonsilent_ranges.append([start_ms, len(audio)])

    valid_ranges = [r for r in nonsilent_ranges if (r[1] - r[0]) >= MIN_SOUND_LEN]

    if len(valid_ranges) < 2:
        print("エラー: 有効な区間が不足しています。")
        return

    # 2. 書き出し処理（librosaによる二次解析を含む）
    print("\n--- 書き出しを開始します ---")
    log_lines = []

    for i in range(1, len(valid_ranges)):
        prev_end = valid_ranges[i - 1][1]
        curr_start = valid_ranges[i][0]
        curr_end = valid_ranges[i][1]

        # -------------------------------------------------------
        # librosaによる精密オフセット計算
        # -------------------------------------------------------
        temp_chunk = audio[curr_start:curr_end]
        vocal_offset = detect_vocal_start_librosa(temp_chunk, mode=DETECTION_MODE)
        # -------------------------------------------------------

        final_silence_end = curr_start + vocal_offset
        final_sound_start = final_silence_end

        # 書き出し
        silence_chunk = audio[prev_end:final_silence_end]
        silence_name = f"silence {i:02d}.wav"
        silence_chunk.export(os.path.join(OUTPUT_DIR, silence_name), format="wav")

        sound_chunk = audio[final_sound_start:curr_end]
        sound_name = f"{i:02d}.wav"
        sound_chunk.export(os.path.join(OUTPUT_DIR, sound_name), format="wav")

        line = f"{sound_name}: {sound_chunk.duration_seconds:.2f}s (Offset: {vocal_offset}ms by librosa-{DETECTION_MODE})"
        print(line)
        log_lines.append(line)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n完了！出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
