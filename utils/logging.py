import logging
import sys
from pathlib import Path
from typing import Union


def setup_logging(name_or_dir: Union[str, Path]) -> logging.Logger:
    """ロギング設定。

    - 文字列を渡した場合: そのモジュール名のロガーを返す（getLogger のラッパー）
    - Path を渡した場合: ファイルハンドラを含むルートロガーを設定し、ルートロガーを返す
    """
    if isinstance(name_or_dir, Path):
        output_dir = name_or_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(output_dir / "experiment.log", encoding="utf-8"),
            ],
        )
        return logging.getLogger()
    else:
        return logging.getLogger(name_or_dir)
