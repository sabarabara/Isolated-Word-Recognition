import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="競技かるた「感じ」の音響学的解析")

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="実験名（結果ディレクトリの名前に使用）",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="音声データのルートディレクトリ"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        required=True,
        help="アノテーションJSONファイルのディレクトリ",
    )
    parser.add_argument(
        "--segment_duration",
        type=str,
        default="0.1",
        help='切り出し長さ（秒）。"full" で全区間使用',
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["1dcnn", "2dcnn", "mock"],
        default="mock",
        help="モデルタイプ",
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["waveform", "melspectrogram"],
        default="melspectrogram",
        help="入力表現タイプ",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./results")

    return parser.parse_args()
