import logging
import sys
from pathlib import Path


def setup_logging(output_dir: Path) -> None:
    """ロギング設定。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "experiment.log", encoding="utf-8"),
        ],
    )
