import os

from pydub import AudioSegment

# --- 設定項目 ---
keep_sec = 0.5  # 抽出する秒数
input_dir = "outputs/wavs/output_06wavs"
output_dir = "outputs/split/0.5s/06wavs"
# ----------------

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        file_path = os.path.join(input_dir, filename)
        audio = AudioSegment.from_file(file_path)

        ms_to_keep = keep_sec * 1000
        extracted_audio = audio[-ms_to_keep:]

        # 保存
        output_path = os.path.join(output_dir, filename)
        extracted_audio.export(output_path, format="wav")
        print(f"Extracted: {filename} (last {keep_sec}s)")

print("\n末尾の抽出がすべて完了しました！")
