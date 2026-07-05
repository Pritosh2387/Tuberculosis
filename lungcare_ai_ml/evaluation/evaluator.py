"""
Model evaluator for LungCare AI.

Runs a full evaluation loop over a DataLoader and returns a rich
results object containing:

- Aggregated scalar metrics (accuracy, F1, AUROC, Dice, IoU, …)
- Per-sample prediction records for downstream analysis
- Confusion matrix (multiclass)
- Calibration statistics (ECE, MCE)
- Optional explainability heatmap generation

Supports classification (binary / multiclass / multilabel) and
segmentation models via task-dispatch.
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

from training.metrics import ClassificationMetrics, SegmentationMetrics

logger = logging.getLogger("lungcare.evaluation.evaluator")


# ─── Result containers ────────────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """
    Aggregated output of a classification evaluation run.

    Attributes:
        metrics: Dict of metric name → scalar value.
        predictions: List of per-sample dicts with keys
            ``'pred_class'``, ``'pred_prob'``, ``'true_class'``, ``'correct'``.
        confusion_matrix: ``(C, C)`` int array (multiclass only).
        class_names: Class label strings in index order.
        calibration: Expected/Maximum calibration error dict.
        eval_time_s: Wall-clock time of the evaluation pass.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    confusion_matrix: np.ndarray | None = None
    class_names: list[str] = field(default_factory=list)
    calibration: dict[str, float] = field(default_factory=dict)
    eval_time_s: float = 0.0


@dataclass
class SegmentationResult:
    """
    Aggregated output of a segmentation evaluation run.

    Attributes:
        metrics: Dict of metric name → scalar value (Dice, IoU, pixel-acc).
        per_sample_dice: Per-image Dice scores.
        eval_time_s: Wall-clock time of the evaluation pass.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    per_sample_dice: list[float] = field(default_factory=list)
    eval_time_s: float = 0.0


# ─── Calibration helpers ──────────────────────────────────────────────────────


def _compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) and Max Calibration Error (MCE).

    Args:
        probs: Confidence of the predicted class, shape ``(N,)``.
        labels: Binary correctness array, shape ``(N,)`` (1 = correct).
        n_bins: Number of equal-width confidence bins.

    Returns:
        Dict with ``'ece'`` and ``'mce'`` in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(probs)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs > lo) & (probs <= hi)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        bin_err = abs(acc - conf)
        ece += (mask.sum() / n) * bin_err
        mce = max(mce, bin_err)

    return {"ece": float(ece), "mce": float(mce)}


# ─── Evaluator ────────────────────────────────────────────────────────────────


class Evaluator:
    """
    Task-agnostic model evaluator for classification and segmentation.

    Args:
        model: Trained PyTorch model.
        loader: DataLoader (yields standard dicts).
        task: ``'binary'``, ``'multiclass'``, ``'multilabel'``,
            or ``'segmentation'``.
        num_classes: Number of classes / labels / segmentation classes.
        device: Inference device.
        class_names: Optional class name list for result annotation.
        threshold: Sigmoid threshold for binary / segmentation prediction.
        amp: Enable AMP for faster GPU inference.
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
        self.model = model.to(device)
        self.loader = loader
        self.task = task
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.threshold = threshold
        self._device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self._amp = amp and self._device_type == "cuda"

    # ─── Public entry points ─────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> ClassificationResult | SegmentationResult:
        """
        Run a full evaluation pass.

        Returns:
            :class:`ClassificationResult` for classification tasks,
            :class:`SegmentationResult` for segmentation.
        """
        if self.task == "segmentation":
            return self._evaluate_segmentation()
        return self._evaluate_classification()

    # ─── Classification ───────────────────────────────────────────────────────

    def _evaluate_classification(self) -> ClassificationResult:
        self.model.eval()
        metrics_agg = ClassificationMetrics(
            num_classes=self.num_classes,
            task=self.task,
            device=self.device,
            class_names=self.class_names,
        )
        metrics_agg.reset()

        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        predictions: list[dict[str, Any]] = []
        t0 = time.time()

        for batch in self.loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            meta = batch.get("metadata", [{}] * images.shape[0])

            ctx = torch.amp.autocast(device_type=self._device_type, enabled=self._amp)
            with ctx:
                logits = self.model(images)

            # Task-specific probability extraction
            if self.task == "multiclass":
                probs = torch.softmax(logits, dim=1)
                pred_cls = probs.argmax(dim=1)
                conf = probs.max(dim=1).values
            else:
                probs = torch.sigmoid(logits)
                if self.task == "binary":
                    pred_cls = (probs.squeeze() >= self.threshold).long()
                    conf = probs.squeeze()
                else:
                    pred_cls = (probs >= self.threshold).long()
                    conf = probs.max(dim=1).values

            metrics_agg.update(probs, labels)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            # Per-sample record
            for i in range(images.shape[0]):
                lbl = labels[i].cpu().numpy()
                prd = pred_cls[i].cpu().numpy()
                predictions.append(
                    {
                        "pred_class": int(prd) if prd.ndim == 0 else prd.tolist(),
                        "pred_prob": probs[i].cpu().numpy().tolist(),
                        "true_class": int(lbl) if lbl.ndim == 0 else lbl.tolist(),
                        "confidence": float(conf[i].item()),
                        "correct": bool(np.array_equal(prd, lbl)),
                        "metadata": meta[i] if isinstance(meta, list) else {},
                    }
                )

        scalar_metrics = metrics_agg.compute()

        # Calibration (multiclass / binary only)
        calibration: dict[str, float] = {}
        if self.task in ("binary", "multiclass") and all_probs:
            probs_np = np.concatenate(all_probs, axis=0)  # (N, C) or (N,)
            labels_np = np.concatenate(all_labels, axis=0)  # (N,) or (N,)
            if self.task == "multiclass":
                pred_cls_np = probs_np.argmax(axis=1)
                conf_np = probs_np.max(axis=1)
                correct_np = (pred_cls_np == labels_np).astype(float)
            else:
                conf_np = probs_np.squeeze()
                correct_np = ((conf_np >= self.threshold) == labels_np).astype(float)
            calibration = _compute_ece(conf_np, correct_np)

        # Confusion matrix (multiclass)
        cm: np.ndarray | None = None
        if self.task == "multiclass" and all_probs:
            import torchmetrics
            cm_metric = torchmetrics.ConfusionMatrix(
                task="multiclass", num_classes=self.num_classes
            )
            preds_t = torch.from_numpy(np.concatenate(all_probs, 0)).argmax(1)
            tgts_t = torch.from_numpy(np.concatenate(all_labels, 0)).long()
            cm_metric.update(preds_t, tgts_t)
            cm = cm_metric.compute().numpy()

        logger.info(
            "Evaluation complete | task=%s | %.1fs | %s",
            self.task, time.time() - t0, scalar_metrics,
        )

        return ClassificationResult(
            metrics=scalar_metrics,
            predictions=predictions,
            confusion_matrix=cm,
            class_names=self.class_names,
            calibration=calibration,
            eval_time_s=round(time.time() - t0, 2),
        )

    # ─── Segmentation ────────────────────────────────────────────────────────

    def _evaluate_segmentation(self) -> SegmentationResult:
        self.model.eval()
        metrics_agg = SegmentationMetrics(
            num_classes=self.num_classes,
            threshold=self.threshold,
            device=self.device,
        )
        metrics_agg.reset()
        per_sample_dice: list[float] = []
        t0 = time.time()

        for batch in self.loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            ctx = torch.amp.autocast(device_type=self._device_type, enabled=self._amp)
            with ctx:
                output = self.model(images)
                if isinstance(output, list):
                    output = output[-1]   # Use finest output for evaluation

            metrics_agg.update(output.detach(), masks)

            # Per-image Dice
            prob = torch.sigmoid(output.squeeze(1))
            pred_bin = (prob >= self.threshold).float()
            tgt = masks.squeeze(1).float()
            inter = (pred_bin * tgt).sum(dim=(1, 2))
            union = pred_bin.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
            dice_batch = ((2.0 * inter + 1.0) / (union + 1.0)).cpu().numpy()
            per_sample_dice.extend(dice_batch.tolist())

        scalar_metrics = metrics_agg.compute()
        logger.info(
            "Segmentation eval complete | Dice=%.4f | IoU=%.4f | %.1fs",
            scalar_metrics.get("dice", 0.0),
            scalar_metrics.get("iou", 0.0),
            time.time() - t0,
        )

        return SegmentationResult(
            metrics=scalar_metrics,
            per_sample_dice=per_sample_dice,
            eval_time_s=round(time.time() - t0, 2),
        )

    # ─── Save helper ─────────────────────────────────────────────────────────

    def save_predictions(
        self,
        result: ClassificationResult,
        output_path: Path,
    ) -> None:
        """
        Save per-sample prediction records to a JSON file.

        Args:
            result: :class:`ClassificationResult` from :meth:`evaluate`.
            output_path: Destination file path (``.json``).
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "metrics": result.metrics,
            "calibration": result.calibration,
            "class_names": result.class_names,
            "eval_time_s": result.eval_time_s,
            "predictions": result.predictions,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Predictions saved → %s", output_path)
