"""
training/trainer.py
────────────────────
Unified training engine for LungCare AI.

Supports both classification and segmentation in a single concrete
class — no abstract base class, no subclasses.

Features
--------
- Automatic Mixed Precision (torch.amp.autocast + GradScaler)
- Gradient clipping by global norm
- Early stopping with configurable patience and metric mode
- TensorBoard logging (per-epoch scalars)
- CSV logging (append-mode, one row per epoch)
- Checkpoint save/load (best model + latest)
- Cosine, step, and plateau LR schedulers

Interview notes
---------------
Why a concrete class instead of an abstract base?
  The abstract BaseTrainer + two subclasses (ClassificationTrainer +
  SegmentationTrainer) added 1200 lines to handle the same training
  loop with minor task-specific differences. Those differences
  (loss function + metric collection) are now constructor arguments,
  which is simpler and more Pythonic.

Why GradScaler only on CUDA?
  AMP reduces float32 to float16 for matmuls, which can underflow
  to zero for small gradients. GradScaler multiplies the loss by a
  large factor before backward(), then divides before optimizer.step()
  to prevent underflow. On CPU, float16 is not natively supported so
  AMP + GradScaler are both disabled.
"""
from __future__ import annotations

import csv
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader


logger = logging.getLogger("lungcare.training.trainer")


# ─── Checkpoint helpers ───────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config_dict: dict[str, Any],
) -> None:
    """
    Save a training checkpoint to *path*.

    Checkpoint dict schema
    ----------------------
    ``model_state``      : model.state_dict()
    ``optimizer_state``  : optimizer.state_dict()
    ``epoch``            : current epoch (int)
    ``metrics``          : val metrics dict
    ``config``           : config snapshot for reproducibility
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch":           epoch,
        "metrics":         metrics,
        "config":          config_dict,
    }, path)
    logger.info("Checkpoint saved → %s (epoch=%d)", path.name, epoch)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint into *model* (and optionally *optimizer*).

    Args:
        path:      Path to the ``.pth`` file.
        model:     Model to load weights into.
        optimizer: If provided, restores optimizer state too.
        device:    Map location for tensors.

    Returns:
        The full checkpoint dict (epoch, metrics, config).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logger.info("Checkpoint loaded ← %s (epoch=%d)", path.name, ckpt.get("epoch", "?"))
    return ckpt


# ─── Early stopping ───────────────────────────────────────────────────────────


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience:  Epochs to wait after the last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode:      ``'max'`` (higher = better) or ``'min'`` (lower = better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.triggered = False

    def __call__(self, value: float) -> bool:
        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ─── CSV logger ───────────────────────────────────────────────────────────────


class _CSVLogger:
    """Append-mode CSV logger — one row per epoch."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path       = path
        self.fieldnames = fieldnames
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames,
                           extrasaction="ignore").writerow(row)


# ─── Trainer ─────────────────────────────────────────────────────────────────


class Trainer:
    """
    Unified training loop for classification and segmentation.

    Args:
        model:          PyTorch model to train.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        criterion:      Loss function.
        optimizer:      Pre-built optimizer (AdamW recommended).
        config:         ``utils.config.Config`` or any object with
                        ``training`` and ``data`` sub-configs.
        task:           ``'multiclass'``, ``'binary'``, or ``'segmentation'``.
        device:         Training device string or ``torch.device``.
        experiment_name: Sub-folder name inside checkpoint_dir and log_dir.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        task: str = "multiclass",
        device: str | torch.device = "cpu",
        experiment_name: str = "run",
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.optimizer    = optimizer
        self.config       = config
        self.task         = task
        self.device       = torch.device(device)
        self._device_type = "cuda" if self.device.type == "cuda" else "cpu"

        tc = config.training   # shorthand

        # ── AMP ───────────────────────────────────────────────────────────────
        self._use_amp = tc.amp and self._device_type == "cuda"
        self._scaler  = (
            torch.amp.GradScaler(self._device_type)
            if self._use_amp else None
        )

        # ── Scheduler ─────────────────────────────────────────────────────────
        self._scheduler = self._build_scheduler(tc.scheduler, tc.epochs)

        # ── Early stopping ────────────────────────────────────────────────────
        self._stopper = EarlyStopping(
            patience=tc.patience,
            mode=tc.monitor_mode,
        )

        # ── Metrics ───────────────────────────────────────────────────────────────
        from training.metrics import ClassificationMetrics, SegmentationMetrics  # lazy
        class_names = getattr(config.data, "class_names", None)
        num_classes  = getattr(config.data, "num_classes", 2)

        if task == "segmentation":
            self._train_metrics: Any = SegmentationMetrics(device=self.device)
            self._val_metrics:   Any = SegmentationMetrics(device=self.device)
        else:
            self._train_metrics = ClassificationMetrics(
                num_classes=num_classes, task=task,
                device=self.device, class_names=class_names,
            )
            self._val_metrics = ClassificationMetrics(
                num_classes=num_classes, task=task,
                device=self.device, class_names=class_names,
            )

        # ── Directories ───────────────────────────────────────────────────────
        self._ckpt_dir = Path(tc.checkpoint_dir) / experiment_name
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_path   = self._ckpt_dir / "best.pth"
        self._latest_path = self._ckpt_dir / "latest.pth"

        # ── TensorBoard ───────────────────────────────────────────────────────
        log_dir = Path(tc.log_dir) / experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info("TensorBoard → %s", log_dir)
        except ImportError:
            logger.warning("tensorboard not installed — skipping TB logging.")

        # ── CSV logger ────────────────────────────────────────────────────────
        self._csv_logger: _CSVLogger | None = None   # lazy init on first epoch

        # ── State ─────────────────────────────────────────────────────────────
        self._best_metric: float = float("-inf")
        logger.info(
            "Trainer | task=%s | device=%s | AMP=%s | epochs=%d",
            task, self.device, self._use_amp, tc.epochs,
        )

    # ── Scheduler factory ─────────────────────────────────────────────────────

    def _build_scheduler(
        self, name: str, epochs: int
    ) -> lr_scheduler.LRScheduler | None:
        if name == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        if name == "step":
            return lr_scheduler.StepLR(
                self.optimizer, step_size=max(1, epochs // 3), gamma=0.1
            )
        if name == "plateau":
            return lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5
            )
        logger.warning("Unknown scheduler '%s' — no scheduler used.", name)
        return None

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> dict[str, Any]:
        """
        Run the full training loop.

        Returns:
            Dict with ``'best_metric'``, ``'best_epoch'``,
            ``'best_checkpoint'``, and final ``'history'`` list.
        """
        tc = self.config.training
        history: list[dict[str, Any]] = []
        best_epoch = 0

        for epoch in range(tc.epochs):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────────────
            train_loss = self._train_epoch()
            train_metrics = self._train_metrics.compute()
            self._train_metrics.reset()

            # ── Validate ──────────────────────────────────────────────────────
            val_loss = self._val_epoch()
            val_metrics = self._val_metrics.compute()
            self._val_metrics.reset()

            # ── Scheduler step ────────────────────────────────────────────────
            monitor_val = val_metrics.get(
                tc.monitor.replace("val_", ""), list(val_metrics.values())[0]
            )
            if self._scheduler:
                if isinstance(self._scheduler, lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(monitor_val)
                else:
                    self._scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            elapsed    = round(time.time() - t0, 2)

            # ── Build epoch record ────────────────────────────────────────────
            record: dict[str, Any] = {
                "epoch":       epoch,
                "train_loss":  round(train_loss, 6),
                "val_loss":    round(val_loss, 6),
                "lr":          current_lr,
                "epoch_time_s": elapsed,
                **{f"train_{k}": round(v, 6) for k, v in train_metrics.items()},
                **{f"val_{k}":   round(v, 6) for k, v in val_metrics.items()},
            }
            history.append(record)

            # ── Log ───────────────────────────────────────────────────────────
            self._log_epoch(epoch, record)

            # ── Checkpoint ────────────────────────────────────────────────────
            is_best = monitor_val > self._best_metric
            if is_best:
                self._best_metric = monitor_val
                best_epoch        = epoch
                save_checkpoint(
                    self._best_path, self.model, self.optimizer,
                    epoch, val_metrics, {}
                )
            save_checkpoint(
                self._latest_path, self.model, self.optimizer,
                epoch, val_metrics, {}
            )

            logger.info(
                "Epoch %03d/%03d | train_loss=%.4f val_loss=%.4f "
                "monitor=%.4f lr=%.2e | %ss",
                epoch + 1, tc.epochs, train_loss, val_loss,
                monitor_val, current_lr, elapsed,
            )

            # ── Early stopping ────────────────────────────────────────────────
            if self._stopper(monitor_val):
                logger.info(
                    "Early stopping triggered at epoch %d "
                    "(patience=%d).", epoch, tc.patience
                )
                break

        if self._tb_writer:
            self._tb_writer.close()

        return {
            "best_metric":     self._best_metric,
            "best_epoch":      best_epoch,
            "best_checkpoint": str(self._best_path),
            "history":         history,
        }

    # ── Train / val epochs ────────────────────────────────────────────────────

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self._device_type, enabled=self._use_amp
            ):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            if self._scaler:
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )
                self.optimizer.step()

            total_loss += loss.item()
            self._update_metrics(self._train_metrics, logits, labels)

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast(
                device_type=self._device_type, enabled=self._use_amp
            ):
                logits = self.model(images)
                loss   = self.criterion(logits, labels)

            total_loss += loss.item()
            self._update_metrics(self._val_metrics, logits, labels)

        return total_loss / max(len(self.val_loader), 1)

    def _update_metrics(
        self,
        metrics: ClassificationMetrics | SegmentationMetrics,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Convert logits to probabilities and update metric accumulators."""
        if self.task == "segmentation":
            metrics.update(logits, labels)  # type: ignore[arg-type]
        elif self.task in ("binary", "multiclass"):
            probs = torch.softmax(logits, dim=1) if self.task == "multiclass" \
                    else torch.sigmoid(logits).squeeze(1)
            metrics.update(probs, labels)   # type: ignore[arg-type]

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_epoch(self, epoch: int, record: dict[str, Any]) -> None:
        # TensorBoard
        if self._tb_writer:
            for key, val in record.items():
                if isinstance(val, float):
                    self._tb_writer.add_scalar(key, val, epoch)

        # CSV (lazy init)
        if self._csv_logger is None:
            csv_path = Path(self.config.training.log_dir) / "training_log.csv"
            self._csv_logger = _CSVLogger(csv_path, list(record.keys()))
        self._csv_logger.log(record)
