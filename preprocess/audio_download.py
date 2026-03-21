import os

import yt_dlp

save_dir = "../data/audio/singer1"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ダウンロードしたいURLのリスト
urls = [
    "https://www.youtube.com/watch?v=BNGa7QsbyeQ",
    "https://www.youtube.com/watch?v=xBzfs7MppWs",
    "https://www.youtube.com/watch?v=cjq4ddzZ1u8",
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
