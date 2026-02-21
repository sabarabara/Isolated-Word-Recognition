import os

import yt_dlp

save_dir = "data/audio/raw/singer2"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ダウンロードしたいURLのリスト
urls = [
    "https://www.youtube.com/watch?v=ZKgqHQfgsT8",
    "https://www.youtube.com/watch?v=qOKQ6H4rK-s",
    "https://www.youtube.com/watch?v=NTylpKj75qQ",
    "https://www.youtube.com/watch?v=aX2Fy77GX0A",
    "https://www.youtube.com/watch?v=g_jHJk92z4U",
]

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": f"{save_dir}/%(title)s.%(ext)s",
    "restrictfilenames": True,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }
    ],
}


with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(urls)

print(f"\nすべて完了！ {save_dir} を確認してください。")
