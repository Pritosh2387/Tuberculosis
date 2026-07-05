"""
Classification trainer for LungCare AI.

Extends :class:`BaseTrainer` with:
- Per-task loss selection (binary / multiclass / multilabel / focal).
- Epoch-level metric collection via :class:`ClassificationMetrics`.
- Optional per-class F1 TensorBoard histogram at the end of training.
- Class-weighted sampling support via ``get_sample_weights()``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.losses import build_classification_loss
from training.metrics import ClassificationMetrics
from training.trainer import BaseTrainer, TrainerConfig

logger = logging.getLogger("lungcare.training.classification")


class ClassificationConfig(TrainerConfig):
    """
    Extended :class:`TrainerConfig` for classification training.

    Extra fields
    ------------
    loss_type: ``'cross_entropy'``, ``'focal'``, ``'bce'``, or ``'label_smoothing'``.
    label_smoothing: Applied when ``loss_type='label_smoothing'``.
    focal_alpha: Alpha for focal loss.
    focal_gamma: Gamma for focal loss.
    class_weights: Optional per-class weights for cross-entropy.
    """

    loss_type: str = "cross_entropy"
    label_smoothing: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    class_weights: list[float] | None = None


class ClassificationTrainer(BaseTrainer):
    """
    Full-featured trainer for multi-disease chest X-ray classification.

    Args:
        model: Classification model (subclass of :class:`BaseClassifier`).
        train_loader: Training :class:`DataLoader` (returns standard dicts).
        val_loader: Validation :class:`DataLoader`.
        config: :class:`ClassificationConfig` instance.
        device: Target device.
        class_names: Optional class name list for per-class metric keys.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ClassificationConfig,
        device: str | torch.device = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__(model, train_loader, val_loader, config, device)
        self.cls_config = config
        self.class_names = class_names or [str(i) for i in range(config.num_classes)]

        # ── Loss function ──────────────────────────────────────────────────────
        cw_tensor: torch.Tensor | None = None
        if config.class_weights:
            cw_tensor = torch.tensor(
                config.class_weights, dtype=torch.float32, device=self.device
            )

        use_focal = config.loss_type == "focal"
        lbl_smooth = config.label_smoothing if config.loss_type == "label_smoothing" else 0.0

        self.criterion = build_classification_loss(
            task=config.task,
            label_smoothing=lbl_smooth,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            use_focal=use_focal,
            pos_weight=cw_tensor if config.task != "multiclass" else None,
        )
        if config.task == "multiclass" and cw_tensor is not None:
            self.criterion = nn.CrossEntropyLoss(
                weight=cw_tensor,
                label_smoothing=lbl_smooth,
            )

        # ── Metrics ────────────────────────────────────────────────────────────
        self._train_metrics = ClassificationMetrics(
            num_classes=config.num_classes,
            task=config.task,
            device=self.device,
            class_names=self.class_names,
        )
        self._val_metrics = ClassificationMetrics(
            num_classes=config.num_classes,
            task=config.task,
            device=self.device,
            class_names=self.class_names,
        )

        # ── Latest epoch metric snapshot ──────────────────────────────────────
        self._last_val_metrics: dict[str, float] = {}

        logger.info(
            "ClassificationTrainer | task=%s | loss=%s | classes=%s",
            config.task, config.loss_type, self.class_names,
        )

    # ─── Step implementations ─────────────────────────────────────────────────

    def _train_step(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        logits = self.model(images)

        if self.cls_config.task == "multiclass":
            loss = self.criterion(logits, labels)
            preds = torch.softmax(logits.detach(), dim=1)
        else:
            loss = self.criterion(logits, labels.float())
            preds = torch.sigmoid(logits.detach())

        self._train_metrics.update(preds, labels)
        return loss

    @torch.no_grad()
    def _val_step(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch["image"].to(self.device)
        labels = batch["label"].to(self.device)

        logits = self.model(images)

        if self.cls_config.task == "multiclass":
            loss = self.criterion(logits, labels)
            preds = torch.softmax(logits, dim=1)
        else:
            loss = self.criterion(logits, labels.float())
            preds = torch.sigmoid(logits)

        self._val_metrics.update(preds, labels)
        return loss

    # ─── Epoch hooks ─────────────────────────────────────────────────────────

    def _pre_train_epoch(self, epoch: int) -> None:
        self._train_metrics.reset()

    def _post_train_epoch(self, epoch: int) -> None:
        pass

    def _compute_train_metrics(self) -> dict[str, float]:
        return self._train_metrics.compute()

    def _compute_val_metrics(self) -> dict[str, float]:
        self._last_val_metrics = self._val_metrics.compute()
        self._val_metrics.reset()
        return self._last_val_metrics

    def _get_monitor_value(self) -> float:
        key = self.config.monitor_metric.removeprefix("val_")
        return self._last_val_metrics.get(key, 0.0)

    # ─── Class-specific helpers ───────────────────────────────────────────────

    def log_per_class_f1(self, epoch: int) -> None:
        """
        Write per-class F1 scores to TensorBoard as a bar chart.

        Call this manually at the end of training for a final report.
        """
        if self._tb_writer is None:
            return
        per_class = {
            k: v
            for k, v in self._last_val_metrics.items()
            if k.startswith("f1_") and k not in ("f1_macro", "f1_weighted")
        }
        for cls_name, f1 in per_class.items():
            self._tb_writer.add_scalar(f"PerClassF1/{cls_name}", f1, global_step=epoch)
