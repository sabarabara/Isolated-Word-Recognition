"""cnn2d 用設定クラス。configs/experiment_configs.yaml からデフォルト値を読み込む。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple
import yaml

_YAML_PATH = Path(__file__).parent / "experiment_configs.yaml"


def _load_yaml_defaults() -> dict:
    with open(_YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class ModelConfig:
    num_classes: int = 100
    dropout_conv: float = 0.2
    dropout_final_conv: float = 0.3
    adaptive_pool_output_2d: Tuple[int, int] = (4, 4)
    fc_hidden_dim: int = 256
    dropout_fc: float = 0.5


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    epochs: int = 10
    grad_clip_max_norm: float = 1.0


@dataclass
class EvaluationConfig:
    top_k_values: List[int] = field(default_factory=lambda: [1, 3, 5])


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    sample_rate: int = 44100


def make_experiment_config(config_dict: dict) -> ExperimentConfig:
    """YAML デフォルト値とランタイムの config_dict をマージして ExperimentConfig を生成する。"""
    defaults = _load_yaml_defaults()

    m = defaults.get("model", {})
    pool = m.get("adaptive_pool_output_2d", [4, 4])
    model = ModelConfig(
        num_classes=config_dict.get("num_classes", m.get("num_classes", 100)),
        dropout_conv=m.get("dropout_conv", 0.2),
        dropout_final_conv=m.get("dropout_final_conv", 0.3),
        adaptive_pool_output_2d=tuple(pool),
        fc_hidden_dim=m.get("fc_hidden_dim", 256),
        dropout_fc=m.get("dropout_fc", 0.5),
    )

    t = defaults.get("training", {})
    training = TrainingConfig(
        learning_rate=config_dict.get("lr", t.get("learning_rate", 1e-3)),
        weight_decay=t.get("weight_decay", 1e-4),
        lr_scheduler_factor=t.get("lr_scheduler_factor", 0.5),
        lr_scheduler_patience=t.get("lr_scheduler_patience", 5),
        early_stopping_patience=t.get("early_stopping_patience", 10),
        early_stopping_min_delta=t.get("early_stopping_min_delta", 0.001),
        epochs=config_dict.get("epochs", t.get("epochs", 10)),
        grad_clip_max_norm=t.get("grad_clip_max_norm", 1.0),
    )

    e = defaults.get("evaluation", {})
    evaluation = EvaluationConfig(
        top_k_values=e.get("top_k_values", [1, 3, 5]),
    )

    return ExperimentConfig(
        model=model,
        training=training,
        evaluation=evaluation,
        sample_rate=defaults.get("sample_rate", 44100),
    )
