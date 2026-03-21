import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
import json
import csv
import time
import logging

from configs.experiment_config import ExperimentConfig
from utils.early_stopping import EarlyStopping
from strategies.evaluation_strategy.strategies.topk.topk_evaluation_strategy import TopKEvaluationStrategy


class _NoOpWriter:
    """TensorBoard なしで動かすための noop ラッパー。"""
    def add_scalars(self, *a, **kw): pass
    def add_scalar(self, *a, **kw): pass
    def close(self): pass

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir

        # ディレクトリ作成
        (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics").mkdir(parents=True, exist_ok=True)

        # 損失関数
        self.criterion = nn.CrossEntropyLoss()

        # オプティマイザ
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # 学習率スケジューラ
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=config.training.lr_scheduler_factor,
            patience=config.training.lr_scheduler_patience,
        )

        # 早期停止
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_min_delta,
        )

        # TensorBoardなしで動かす
        self.writer = _NoOpWriter()

        # CSVロガー
        self.csv_path = output_dir / "metrics" / "training_log.csv"
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc_top1",
                "val_loss", "val_acc_top1", "val_acc_top3", "val_acc_top5",
                "learning_rate", "elapsed_sec"
            ])

    def train(self) -> Dict:
        """
        全エポックの学習を実行する。

        Returns
        -------
        best_metrics : Dict
            最良エポックのメトリクス
        """
        best_val_acc = 0.0
        best_metrics = {}

        for epoch in range(1, self.config.training.epochs + 1):
            start_time = time.time()

            # 学習
            train_loss, train_acc = self._train_one_epoch()

            # 検証（valが空の場合はtrainデータで検証）
            if len(self.val_loader.dataset) > 0:
                val_loss, val_metrics = self._validate()
            else:
                logger.warning("検証セットが空のため、訓練セットで評価します")
                val_loss, val_metrics = self._validate_on_train()

            elapsed = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch}/{self.config.training.epochs} "
                f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% "
                f"| Val Loss: {val_loss:.4f} "
                f"Top-1: {val_metrics['top_1']:.2f}% "
                f"Top-3: {val_metrics['top_3']:.2f}% "
                f"Top-5: {val_metrics['top_5']:.2f}% "
                f"| LR: {current_lr:.2e} "
                f"| Time: {elapsed:.1f}s"
            )

            # TensorBoard記録
            self.writer.add_scalars("Loss", {
                "train": train_loss, "val": val_loss
            }, epoch)
            self.writer.add_scalars("Accuracy/Top-1", {
                "train": train_acc, "val": val_metrics["top_1"]
            }, epoch)
            self.writer.add_scalar(
                "Accuracy/Val_Top-3", val_metrics["top_3"], epoch
            )
            self.writer.add_scalar(
                "Accuracy/Val_Top-5", val_metrics["top_5"], epoch
            )
            self.writer.add_scalar("LearningRate", current_lr, epoch)

            # CSV記録
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, f"{train_loss:.6f}", f"{train_acc:.4f}",
                    f"{val_loss:.6f}", f"{val_metrics['top_1']:.4f}",
                    f"{val_metrics['top_3']:.4f}",
                    f"{val_metrics['top_5']:.4f}",
                    f"{current_lr:.2e}", f"{elapsed:.2f}"
                ])

            # 学習率スケジューラ更新
            self.scheduler.step(val_metrics["top_1"])

            # ベストモデル保存
            if val_metrics["top_1"] > best_val_acc:
                best_val_acc = val_metrics["top_1"]
                best_metrics = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()}
                }
                self._save_checkpoint(epoch, is_best=True)

            # 定期チェックポイント（10エポックごと）
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, is_best=False)

            # 早期停止判定
            if self.early_stopping(val_metrics["top_1"]):
                logger.info(f"早期停止: Epoch {epoch}")
                break

        self.writer.close()

        # 最終結果の保存
        with open(
            self.output_dir / "metrics" / "best_metrics.json", "w"
        ) as f:
            json.dump(best_metrics, f, indent=2, ensure_ascii=False)

        return best_metrics

    def _train_one_epoch(self) -> tuple:
        """1エポックの学習。"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels, _metadata) in enumerate(
            self.train_loader
        ):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()

            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.grad_clip_max_norm,
            )

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(dim=1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def _validate(self) -> tuple:
        """検証。"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _metadata in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        total = all_labels.size(0)

        avg_loss = total_loss / total

        metrics = TopKEvaluationStrategy(
            k_list=self.config.evaluation.top_k_values,
        ).evaluate(all_outputs, all_labels)

        return avg_loss, metrics

    def _validate_on_train(self):
        """訓練データで検証（検証セットが空の場合の代替）"""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, _ in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        total = all_labels.size(0)

        avg_loss = total_loss / total

        metrics = TopKEvaluationStrategy(
            k_list=self.config.evaluation.top_k_values,
        ).evaluate(all_outputs, all_labels)

        return avg_loss, metrics

    def _save_checkpoint(self, epoch: int, is_best: bool):
        """チェックポイント保存。"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
        }

        if is_best:
            path = self.output_dir / "checkpoints" / "best_model.pt"
        else:
            path = self.output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        logger.info(f"チェックポイント保存: {path}")
        