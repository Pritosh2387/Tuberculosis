"""
Metric collections for LungCare AI training and evaluation.

All metric classes follow a three-call contract compatible with
:class:`torch.utils.data.DataLoader` iteration:

.. code-block:: python

    metrics = ClassificationMetrics(num_classes=6, task="multiclass")
    metrics.reset()
    for batch in loader:
        metrics.update(preds, targets)
    result: dict = metrics.compute()

Supported tasks
---------------
- ``'binary'``     — Accuracy, F1, AUROC, Precision, Recall, Specificity
- ``'multiclass'`` — Accuracy, F1 (macro + weighted), AUROC, Confusion Matrix
- ``'multilabel'`` — Accuracy, F1 (macro), AUROC (per label), Exact Match
- Segmentation     — Dice, IoU (Jaccard), Pixel Accuracy
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torchmetrics

logger = logging.getLogger("lungcare.training.metrics")


# ─── Classification metrics ───────────────────────────────────────────────────


class ClassificationMetrics:
    """
    torchmetrics-backed metric collection for classification tasks.

    Args:
        num_classes: Number of classes (labels for multilabel).
        task: ``'binary'``, ``'multiclass'``, or ``'multilabel'``.
        device: Tensor device.
        class_names: Optional list of class names for per-class reporting.
    """

    def __init__(
        self,
        num_classes: int,
        task: str,
        device: str | torch.device = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.task = task
        self.device = device
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self._collection = self._build(task, num_classes, device)

    def _build(
        self,
        task: str,
        num_classes: int,
        device: str | torch.device,
    ) -> torchmetrics.MetricCollection:
        if task == "binary":
            metrics: dict[str, torchmetrics.Metric] = {
                "accuracy": torchmetrics.Accuracy(task="binary"),
                "f1": torchmetrics.F1Score(task="binary"),
                "auroc": torchmetrics.AUROC(task="binary"),
                "precision": torchmetrics.Precision(task="binary"),
                "recall": torchmetrics.Recall(task="binary"),
                "specificity": torchmetrics.Specificity(task="binary"),
            }

        elif task == "multiclass":
            metrics = {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                ),
                "f1_macro": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                ),
                "f1_weighted": torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    average="weighted",
                ),
                "auroc": torchmetrics.AUROC(
                    task="multiclass", num_classes=num_classes
                ),
                "precision_macro": torchmetrics.Precision(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                ),
                "recall_macro": torchmetrics.Recall(
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                ),
            }
            # Separate per-class F1 for clinical interpretability
            for i in range(num_classes):
                name = self.class_names[i] if self.class_names else str(i)
                metrics[f"f1_{name}"] = torchmetrics.F1Score(
                    task="multiclass",
                    num_classes=num_classes,
                    average="none",
                )

        elif task == "multilabel":
            metrics = {
                "accuracy": torchmetrics.Accuracy(
                    task="multilabel", num_labels=num_classes
                ),
                "f1_macro": torchmetrics.F1Score(
                    task="multilabel",
                    num_labels=num_classes,
                    average="macro",
                ),
                "auroc": torchmetrics.AUROC(
                    task="multilabel", num_labels=num_classes
                ),
                "exact_match": torchmetrics.ExactMatch(
                    task="multilabel", num_labels=num_classes
                ),
                "precision_macro": torchmetrics.Precision(
                    task="multilabel",
                    num_labels=num_classes,
                    average="macro",
                ),
                "recall_macro": torchmetrics.Recall(
                    task="multilabel",
                    num_labels=num_classes,
                    average="macro",
                ),
            }
        else:
            raise ValueError(f"Unknown task: '{task}'")

        return torchmetrics.MetricCollection(metrics).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate one batch of predictions.

        Args:
            preds: Class probabilities or sigmoid scores.
                - ``'binary'``: ``(B,)`` positive-class probability.
                - ``'multiclass'``: ``(B, num_classes)`` softmax scores.
                - ``'multilabel'``: ``(B, num_labels)`` sigmoid scores.
            targets: Ground truth.
                - ``'binary'``: ``(B,)`` long tensor (0 or 1).
                - ``'multiclass'``: ``(B,)`` class index long tensor.
                - ``'multilabel'``: ``(B, num_labels)`` long tensor.
        """
        self._collection.update(preds.to(self.device), targets.to(self.device))

    def compute(self) -> dict[str, float]:
        """
        Compute and return all accumulated metrics.

        For per-class F1 in multiclass mode, the returned dict contains
        an extra key per class (e.g. ``'f1_Tuberculosis': 0.87``).

        Returns:
            Dict of metric name → scalar float value.
        """
        raw = self._collection.compute()
        result: dict[str, float] = {}

        for k, v in raw.items():
            tensor_val = v.cpu()
            if tensor_val.ndim == 0:
                result[k] = float(tensor_val.item())
            elif tensor_val.ndim == 1 and k.startswith("f1_") and self.task == "multiclass":
                # Per-class F1 vector → expand to named keys
                for idx, cls_name in enumerate(self.class_names):
                    result[f"f1_{cls_name}"] = float(tensor_val[idx].item())
            else:
                result[k] = float(tensor_val.mean().item())

        return result

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._collection.reset()

    def to(self, device: str | torch.device) -> "ClassificationMetrics":
        """Move all metrics to *device*."""
        self._collection = self._collection.to(device)
        self.device = device
        return self

    def get_confusion_matrix(self) -> torch.Tensor | None:
        """
        Return the confusion matrix for multiclass tasks.

        Returns ``None`` for binary / multilabel tasks.
        """
        if self.task != "multiclass":
            return None
        cm = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.num_classes
        ).to(self.device)
        # Note: caller is responsible for updating cm separately
        return cm


# ─── Segmentation metrics ─────────────────────────────────────────────────────


class SegmentationMetrics:
    """
    Metric collection for binary / multiclass segmentation.

    Args:
        num_classes: Number of foreground classes (1 = binary segmentation).
        threshold: Decision threshold for sigmoid predictions.
        device: Tensor device.
    """

    def __init__(
        self,
        num_classes: int = 1,
        threshold: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.threshold = threshold
        self.device = device

        # NOTE: ``torchmetrics.Dice`` was removed in torchmetrics>=1.6.  For
        # segmentation the Dice coefficient is identical to the F1 score
        # (2·TP / (2·TP + FP + FN)), so F1Score is used as a drop-in that keeps
        # the ``dice`` output key and is available across torchmetrics versions.
        if num_classes == 1:
            tm_task = "binary"
            self._dice = torchmetrics.F1Score(task="binary").to(device)
            self._iou = torchmetrics.JaccardIndex(task="binary").to(device)
            self._acc = torchmetrics.Accuracy(task="binary").to(device)
        else:
            tm_task = "multiclass"
            self._dice = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ).to(device)
            self._iou = torchmetrics.JaccardIndex(
                task="multiclass", num_classes=num_classes
            ).to(device)
            self._acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            ).to(device)

        self._tm_task = tm_task

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Accumulate one batch.

        Args:
            logits: Raw model output ``(B, 1, H, W)`` or ``(B, C, H, W)``.
            targets: Binary / class masks ``(B, 1, H, W)`` or ``(B, H, W)``.
        """
        if self.num_classes == 1:
            preds_prob = torch.sigmoid(logits.squeeze(1))
            preds_bin = (preds_prob >= self.threshold).long()
            tgt = targets.squeeze(1).long().to(self.device)
        else:
            preds_bin = logits.argmax(dim=1)
            tgt = targets.squeeze(1).long().to(self.device)

        preds_bin = preds_bin.to(self.device)
        self._dice.update(preds_bin, tgt)
        self._iou.update(preds_bin, tgt)
        self._acc.update(preds_bin, tgt)

    def compute(self) -> dict[str, float]:
        """Return accumulated metrics as a dict."""
        return {
            "dice": float(self._dice.compute().item()),
            "iou": float(self._iou.compute().item()),
            "pixel_accuracy": float(self._acc.compute().item()),
        }

    def reset(self) -> None:
        """Reset all accumulated state."""
        self._dice.reset()
        self._iou.reset()
        self._acc.reset()

    def to(self, device: str | torch.device) -> "SegmentationMetrics":
        """Move all metrics to *device*."""
        self._dice = self._dice.to(device)
        self._iou = self._iou.to(device)
        self._acc = self._acc.to(device)
        self.device = device
        return self
