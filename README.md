
## 📁 ディレクトリ構成


```text
karuta_project/
├── main.py                     # メイン実行エントリポイント
├── factories.py                # モデルや戦略インスタンスの生成ファクトリ
├── configs/                    # 設定ファイル管理
│   └── experiment_configs.yaml # 学習パラメータ、モデル選択、パス設定
├── args/                       # 実行引数管理
│   └── arg_parser.py           # コマンドライン引数の定義
├── data/                       # データストレージ
│   ├── anotation_data/         # 教師ラベルデータ（アノテーション）
│   ├── audio/                  # 生音声データ（WAV, MP3等）
│   ├── text/                   # テキスト関連データ
│   └── outputs/                # 中間処理生成物
│       ├── segment/            # VAD等によるセグメント済み音声
│       ├── split/              # 分割済みデータ（intervals/wavs）
│       └── stt/                # Speech-to-Text（文字起こし）結果
├── preprocess/                 # 前処理パイプライン
│   ├── audio_download.py       # 音声ソースの取得
│   ├── audio_segmentaion.py    # 音声区間検出
│   ├── audio_speak_to_text.py  # 音声のテキスト化処理
│   ├── audio_split/            # librosa/pydubを用いた断片化処理
│   ├── augmentation.py         # データ拡張（ノイズ付加、伸縮等）
│   ├── create_annotation.py    # 学習用ラベルファイルの作成
│   └── feature_extraction.py   # メルスペクトログラム等の特徴量抽出
├── model_strategy/             # 学習アルゴリズム（Strategy）
│   ├── model_strategy.py       # 基本インターフェース定義
│   ├── context.py              # 学習実行コンテキスト
│   └── strategys/              # 各種モデル実装
│       ├── AST/                # Audio Spectrogram Transformer
│       ├── GRU/                # Gated Recurrent Unit
│       ├── LSTM/               # Long Short-Term Memory
│       ├── cnn1d/              # 1次元CNN（波形ベース）
│       │   ├── cnn1d_strategy.py # CNN1D専用の戦略ロジック
│       │   ├── dataset.py        # CNN1D用データローダー
│       │   ├── evaluator.py      # モデル評価ロジック
│       │   ├── model.py          # ネットワーク定義（PyTorch等）
│       │   └── trainer.py        # 学習ループの実装
│       └── cnn2d/              # 2次元CNN（スペクトログラムベース）
├── evaluation_strategy/        # 評価アルゴリズム
│   ├── evaluation_strategy.py  # 評価の基底クラス
│   ├── context.py              # 評価実行コンテキスト
│   └── strategies/             # 具体的な評価手法
│       ├── confusion_matrix/   # 混合行列による誤分類分析
│       └── topk/               # Top-k Accuracyの算出
├── utils/                      # 共通ユーティリティ
│   ├── seed.py                 # 再現性のための乱数シード固定
│   └── analysis/               # 統計検定（T検定、有意差分析等）
└── results/                    # 実験結果の出力先（Model, Logs, Plots）