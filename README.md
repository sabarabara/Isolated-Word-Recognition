
## 📁 ディレクトリ構成


karuta_project/
├── README.md
├── __init__.py
├── factories.py                # インスタンス生成（Factory Pattern）
├── main.py                     # メイン実行スクリプト
├── requirements.txt            # 依存ライブラリ
├── configs/                    # 設定ファイル
│   ├── __init__.py
│   ├── config.yaml             # Hydra 実行設定
│   └── experiment_configs.yaml
├── data/                       # データディレクトリ
│   ├── __init__.py
│   ├── anotation_data/         # アノテーション（正解ラベル）
│   │   └── __init__.py
│   ├── audio/                  # 生音声データ
│   │   └── __init__.py
│   ├── text/                   # テキスト・スクリプト等
│   │   └── __init__.py
│   └── outputs/                # 各工程の出力
│       ├── __init__.py
│       ├── segment/            # セグメンテーション結果
│       │   └── __init__.py
│       ├── split/              # 分割データ
│       │   ├── __init__.py
│       │   ├── intervals/      # 区間情報
│       │   │   └── __init__.py
│       │   └── wavs/           # 分割済みWAV
│       │       └── __init__.py
│       └── stt/                # Speech-to-Text 結果
│           └── __init__.py
├── preprocess/                 # 前処理パイプライン
│   ├── __init__.py
│   ├── audio_download.py
│   ├── audio_segmentaion.py
│   ├── audio_speak_to_text.py
│   ├── augmentation.py         # データ拡張
│   ├── create_annotation.py
│   ├── feature_extraction.py   # 特徴量抽出（MFCC等）
│   └── audio_split/            # 分割処理ライブラリ別
│       ├── __init__.py
│       ├── audio_split_librosa.py
│       └── audio_split_pydup.py
├── strategies/                 # 戦略（Strategy Pattern）実装
│   ├── __init__.py
│   ├── registry.py             # クラス登録用
│   ├── evaluation_strategy/    # 評価ロジック
│   │   ├── __init__.py
│   │   ├── composite.py        # 複数評価の統合
│   │   ├── context.py          # 実行コンテキスト
│   │   ├── evaluation_strategy.py # 基底クラス
│   │   └── strategies/         # 具体的な評価手法
│   │       ├── __init__.py
│   │       ├── confusion_matrix/
│   │       │   ├── __init__.py
│   │       │   └── confusion_matrix_evaluation_strategy.py
│   │       ├── mock/           # テスト用
│   │       │   ├── __init__.py
│   │       │   └── simple_eval_strategy.py
│   │       └── topk/
│   │           ├── __init__.py
│   │           └── topk_evaluation_strategy.py
│   └── model_strategy/         # モデル・学習ロジック
│       ├── __init__.py
│       ├── context.py
│       ├── model_strategy.py   # 基底クラス
│       └── strategys/          # モデル別実装
│           ├── AST/            # Audio Spectrogram Transformer
│           ├── GRU/            # Gated Recurrent Unit
│           ├── LSTM/           # Long Short-Term Memory
│           ├── cnn1d/          # 1D-CNN 実装一式
│           │   ├── __init__.py
│           │   ├── cnn1d_strategy.py
│           │   ├── dataset.py
│           │   ├── evaluator.py
│           │   ├── model.py
│           │   └── trainer.py
│           ├── cnn2d/          # 2D-CNN
│           └── mock/           # テスト用
│               └── simple_model_strategy.py
├── utils/                      # 共通ツール
│   ├── __init__.py
│   ├── logging.py              # ログ出力
│   ├── seed.py                 # 乱数固定
│   └── analysis/               # 分析用ツール
│       ├── __init__.py
│       └── statistical_tests.py # 統計検定
└── results/                    # 実験結果保存用
    └── __init__.py


python main.py experiment_name=test data_dir=data annotation_dir=data/annotation_data