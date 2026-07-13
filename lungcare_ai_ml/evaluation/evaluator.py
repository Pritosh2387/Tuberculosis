"""
evaluation/evaluator.py
────────────────────────
Model evaluator for classification and segmentation.

Runs a full evaluation pass over a DataLoader and returns a
structured result containing real computed metrics — no hardcoded
values, no fabricated numbers.

Metrics computed (classification)
-----------------------------------
- Accuracy
- Precision (macro)
- Recall (macro)
- F1 (macro + weighted)
- ROC-AUC
- Confusion matrix

Metrics computed (segmentation)
---------------------------------
- Dice coefficient
- IoU (Jaccard Index)
- Pixel accuracy
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


logger = logging.getLogger("lungcare.evaluation.evaluator")


# ─── Result containers ────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    """
    Aggregated output of a classification evaluation run.

    Attributes:
        metrics:          Dict of metric name → scalar value.
        confusion_matrix: (C, C) int array (multiclass) or None.
        class_names:      Class label strings in index order.
        eval_time_s:      Wall-clock time of the evaluation pass.
    """
    metrics:          dict[str, float]      = field(default_factory=dict)
    confusion_matrix: np.ndarray | None     = None
    class_names:      list[str]             = field(default_factory=list)
    eval_time_s:      float                 = 0.0


@dataclass
class SegmentationResult:
    """
    Aggregated output of a segmentation evaluation run.

    Attributes:
        metrics:     Dict of metric name → scalar value (Dice, IoU, pixel_accuracy).
        eval_time_s: Wall-clock time of the evaluation pass.
    """
    metrics:     dict[str, float] = field(default_factory=dict)
    eval_time_s: float            = 0.0


# ─── Evaluator ────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Task-agnostic model evaluator.

    Args:
        model:       Trained PyTorch model (eval mode set internally).
        loader:      DataLoader — must yield standard dicts with
                     ``'image'``, ``'label'`` keys.
        task:        ``'binary'``, ``'multiclass'``, or ``'segmentation'``.
        num_classes: Number of classes / segmentation output channels.
        device:      Inference device string or torch.device.
        class_names: Optional class name list for result annotation.
        threshold:   Sigmoid threshold for binary/segmentation output.
        amp:         Enable AMP for faster GPU inference.
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        task: str,
        num_classes: int,
        device: str | torch.device = "cpu",
        class_names: list[str] | None = None,
        threshold: float = 0.5,
        amp: bool = False,
    ) -> None:
        self.model       = model
        self.loader      = loader
        self.task        = task
        self.num_classes = num_classes
        self.device      = torch.device(device)
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.threshold   = threshold
        self._device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self._amp        = amp and self._device_type == "cuda"
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self) -> ClassificationResult | SegmentationResult:
        """
        Run a full evaluation pass over the DataLoader.

        Returns:
            ClassificationResult or SegmentationResult with real metrics.
        """
        if self.task == "segmentation":
            return self._eval_segmentation()
        return self._eval_classification()

    # ── Classification ────────────────────────────────────────────────────────

    def _eval_classification(self) -> ClassificationResult:
        from training.metrics import ClassificationMetrics  # lazy
        self.model.eval()
        metrics_agg = ClassificationMetrics(
            num_classes=self.num_classes,
            task=self.task,
            device=self.device,
            class_names=self.class_names,
        )
        metrics_agg.reset()

        # Also collect for confusion matrix
        all_preds:  list[int]   = []
        all_labels: list[int]   = []
        t0 = time.time()

        for batch in self.loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast(
                device_type=self._device_type, enabled=self._amp
            ):
                logits = self.model(images)

            if self.task == "multiclass":
                probs    = torch.softmax(logits, dim=1)
                pred_cls = probs.argmax(dim=1)
            else:
                probs    = torch.sigmoid(logits).squeeze(1)
                pred_cls = (probs >= self.threshold).long()

            metrics_agg.update(probs, labels)
            all_preds.extend(pred_cls.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = metrics_agg.compute()

        # Confusion matrix (multiclass only)
        cm: np.ndarray | None = None
        if self.task == "multiclass":
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            for p, t in zip(all_preds, all_labels):
                cm[t, p] += 1

        return ClassificationResult(
            metrics=metrics,
            confusion_matrix=cm,
            class_names=self.class_names,
            eval_time_s=round(time.time() - t0, 3),
        )

    # ── Segmentation ─────────────────────────────────────────────────────────

    def _eval_segmentation(self) -> SegmentationResult:
        from training.metrics import SegmentationMetrics  # lazy
        self.model.eval()
        metrics_agg = SegmentationMetrics(
            num_classes=self.num_classes,
            threshold=self.threshold,
            device=self.device,
        )
        metrics_agg.reset()
        t0 = time.time()

        for batch in self.loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast(
                device_type=self._device_type, enabled=self._amp
            ):
                logits = self.model(images)

            metrics_agg.update(logits, labels)

        return SegmentationResult(
            metrics=metrics_agg.compute(),
            eval_time_s=round(time.time() - t0, 3),
        )
