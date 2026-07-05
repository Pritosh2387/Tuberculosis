"""
Abstract base trainer for LungCare AI.

Provides a production-grade training loop with:

- Single-GPU and :class:`~torch.nn.DataParallel` multi-GPU support.
- Automatic Mixed Precision (``torch.amp``).
- Gradient accumulation across N steps before an optimizer update.
- Gradient clipping by global norm.
- Early stopping with configurable patience and mode.
- Full checkpoint resume: model, optimizer, scheduler, scaler.
- CSV logging per epoch.
- Optional TensorBoard logging (requires ``tensorboard`` package).
"""

from __future__ import annotations

import csv
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from training.schedulers import build_scheduler, is_step_scheduler
from utils.checkpoint import CheckpointManager

logger = logging.getLogger("lungcare.training.trainer")


# ─── Pydantic config models ───────────────────────────────────────────────────


class OptimizerConfig(BaseModel):
    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9


class SchedulerConfig(BaseModel):
    name: str = "warmup_cosine"
    T_max: int = 100
    eta_min: float = 1e-6
    warmup_epochs: int = 5
    eta_min_ratio: float = 0.0
    T_0: int = 10
    T_mult: int = 2
    step_size: int = 30
    gamma: float = 0.1
    milestones: list[int] = []
    factor: float = 0.5
    patience: int = 10
    mode: str = "max"
    max_lr: float = 1e-3
    pct_start: float = 0.3


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 15
    min_delta: float = 1e-4
    mode: str = "max"   # 'max' for metrics like F1/Dice, 'min' for loss


class TrainerConfig(BaseModel):
    task: str = "multiclass"
    num_classes: int = 6
    epochs: int = 100
    amp: bool = True
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    data_parallel: bool = False
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")
    experiment_name: str = "experiment"
    save_every_n_epochs: int = 5
    resume_from: Path | None = None
    monitor_metric: str = "val_f1_macro"
    monitor_mode: str = "max"   # 'max' → higher is better
    top_k_checkpoints: int = 3
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)


# ─── Utilities ────────────────────────────────────────────────────────────────


class EarlyStopping:
    """
    Monitors a scalar metric and triggers when no improvement is seen
    for *patience* consecutive epochs.

    Args:
        patience: Epochs to wait after last improvement.
        min_delta: Minimum absolute change to be considered an improvement.
        mode: ``'max'`` (higher = better) or ``'min'`` (lower = better).
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter: int = 0
        self.best_value: float = float("-inf") if mode == "max" else float("inf")
        self.triggered: bool = False

    def __call__(self, value: float) -> bool:
        """
        Update state with the latest metric value.

        Returns:
            ``True`` if training should stop.
        """
        if self._is_improvement(value):
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.triggered = True

        return self.triggered

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_value + self.min_delta
        return value < self.best_value - self.min_delta

    def state_dict(self) -> dict[str, Any]:
        return {
            "counter": self.counter,
            "best_value": self.best_value,
            "triggered": self.triggered,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.counter = state["counter"]
        self.best_value = state["best_value"]
        self.triggered = state["triggered"]


class _CSVLogger:
    """Append-mode CSV logger; creates the file and header on first call."""

    def __init__(self, filepath: Path, fieldnames: list[str]) -> None:
        self.filepath = filepath
        self.fieldnames = fieldnames
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict[str, Any]) -> None:
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
            writer.writerow(row)


# ─── Abstract base trainer ────────────────────────────────────────────────────


class BaseTrainer(ABC):
    """
    Abstract training loop engine.

    Subclasses must implement :meth:`_train_step`, :meth:`_val_step`,
    and :meth:`_get_monitor_value`.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        config: :class:`TrainerConfig` instance.
        device: Target device string or :class:`torch.device`.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self._device_type: str = "cuda" if self.device.type == "cuda" else "cpu"
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── Model setup ───────────────────────────────────────────────────────
        if config.data_parallel and torch.cuda.device_count() > 1:
            logger.info("Using DataParallel across %d GPUs.", torch.cuda.device_count())
            self.model: nn.Module = nn.DataParallel(model)
        else:
            self.model = model
        self.model.to(self.device)

        # ── Optimiser ─────────────────────────────────────────────────────────
        self.optimizer = self._build_optimizer(config.optimizer)

        # ── AMP ───────────────────────────────────────────────────────────────
        self._use_amp = config.amp and self._device_type == "cuda"
        self.scaler: torch.amp.GradScaler | None = (
            torch.amp.GradScaler(self._device_type) if self._use_amp else None
        )

        # ── Scheduler ─────────────────────────────────────────────────────────
        total_steps = len(train_loader) * config.epochs
        self.scheduler = build_scheduler(
            self.optimizer,
            config.scheduler,
            total_steps=total_steps,
            total_epochs=config.epochs,
        )
        self._per_step_scheduler = is_step_scheduler(self.scheduler)

        # ── Early stopping ────────────────────────────────────────────────────
        es = config.early_stopping
        self.early_stopper = EarlyStopping(
            patience=es.patience,
            min_delta=es.min_delta,
            mode=es.mode,
        )

        # ── Checkpoint manager ────────────────────────────────────────────────
        ckpt_dir = config.checkpoint_dir / config.experiment_name
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            monitor=config.monitor_metric,
            mode=config.monitor_mode,
            top_k=config.top_k_checkpoints,
        )

        # ── Logging ───────────────────────────────────────────────────────────
        log_dir = config.log_dir / config.experiment_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self._csv_logger: _CSVLogger | None = None  # created lazily

        self._tb_writer: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info("TensorBoard writer → %s", log_dir)
        except ImportError:
            logger.warning("tensorboard not installed — skipping TensorBoard logging.")

        # ── State ─────────────────────────────────────────────────────────────
        self.start_epoch: int = 0
        self.global_step: int = 0

        # ── Resume ────────────────────────────────────────────────────────────
        if config.resume_from:
            self.start_epoch = self._resume(Path(config.resume_from))

        logger.info(
            "BaseTrainer | device=%s | AMP=%s | grad_accum=%d | epochs=%d",
            self.device, self._use_amp, config.grad_accum_steps, config.epochs,
        )

    # ─── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def _train_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Process one batch in training mode.

        Args:
            batch: Dict from the DataLoader (image, label, mask, metadata).

        Returns:
            Scalar loss tensor.  Do **not** call ``.backward()`` here;
            the base trainer handles gradient accumulation.
        """

    @abstractmethod
    def _val_step(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Process one batch in evaluation mode.

        Args:
            batch: Dict from the DataLoader.

        Returns:
            Scalar loss tensor (no grad required).
        """

    @abstractmethod
    def _get_monitor_value(self) -> float:
        """
        Extract the scalar value of ``config.monitor_metric`` from the
        latest computed validation metrics.  Used by early stopping and
        checkpoint manager.
        """

    # ─── Main training loop ───────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run the full training loop from :attr:`start_epoch` to ``config.epochs``.
        """
        logger.info("Starting training from epoch %d.", self.start_epoch)
        for epoch in range(self.start_epoch, self.config.epochs):
            t0 = time.time()

            # ── Train ─────────────────────────────────────────────────────────
            self._pre_train_epoch(epoch)
            train_loss = self._train_epoch(epoch)
            train_metrics = self._compute_train_metrics()
            self._post_train_epoch(epoch)

            # ── Validate ──────────────────────────────────────────────────────
            val_loss = self._val_epoch(epoch)
            val_metrics = self._compute_val_metrics()

            # ── Scheduler step ────────────────────────────────────────────────
            if self.scheduler and not self._per_step_scheduler:
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    monitor_val = self._get_monitor_value()
                    self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()

            # ── Current LR ────────────────────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ── Logging ───────────────────────────────────────────────────────
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "lr": current_lr,
                "epoch_time_s": round(time.time() - t0, 2),
                **{f"train_{k}": round(v, 6) for k, v in train_metrics.items()},
                **{f"val_{k}": round(v, 6) for k, v in val_metrics.items()},
            }
            self._log_epoch(epoch, epoch_metrics)
            self._log_to_tb(epoch, epoch_metrics)

            # ── Checkpoint ────────────────────────────────────────────────────
            monitor_val = self._get_monitor_value()
            self.checkpoint_manager.save(
                epoch=epoch,
                metric_value=monitor_val,
                model=self._unwrap_model(),
                optimizer=self.optimizer,
                extra={
                    "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "early_stopping_state": self.early_stopper.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "config": self.config.model_dump(),
                },
            )

            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %s=%.4f | lr=%.2e | %.1fs",
                epoch + 1, self.config.epochs,
                train_loss, val_loss,
                self.config.monitor_metric, monitor_val,
                current_lr,
                time.time() - t0,
            )

            # ── Early stopping ─────────────────────────────────────────────────
            if self.config.early_stopping.enabled:
                if self.early_stopper(monitor_val):
                    logger.info(
                        "Early stopping triggered at epoch %d (best=%.4f).",
                        epoch + 1, self.early_stopper.best_value,
                    )
                    break

        logger.info("Training complete.")
        if self._tb_writer:
            self._tb_writer.close()

    # ─── Epoch loops ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            ctx = torch.amp.autocast(
                device_type=self._device_type, enabled=self._use_amp
            )
            with ctx:
                loss = self._train_step(batch)
                loss = loss / self.config.grad_accum_steps

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            is_accum = (step + 1) % self.config.grad_accum_steps == 0
            is_last = (step + 1) == len(self.train_loader)

            if is_accum or is_last:
                if self.config.grad_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                if self._per_step_scheduler and self.scheduler:
                    self.scheduler.step()

            total_loss += loss.item() * self.config.grad_accum_steps

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            loss = self._val_step(batch)
            total_loss += loss.item()

        return total_loss / max(len(self.val_loader), 1)

    # ─── Hooks (override for custom behaviour) ────────────────────────────────

    def _pre_train_epoch(self, epoch: int) -> None:
        """Called at the start of each training epoch."""

    def _post_train_epoch(self, epoch: int) -> None:
        """Called at the end of each training epoch."""

    def _compute_train_metrics(self) -> dict[str, float]:
        """Return training metrics after one epoch.  Override in subclass."""
        return {}

    def _compute_val_metrics(self) -> dict[str, float]:
        """Return validation metrics after one epoch.  Override in subclass."""
        return {}

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _build_optimizer(self, cfg: OptimizerConfig) -> optim.Optimizer:
        params = self._unwrap_model().parameters()
        name = cfg.name.lower()
        if name == "adamw":
            return optim.AdamW(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )
        elif name == "adam":
            return optim.Adam(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
                eps=cfg.eps,
            )
        elif name == "sgd":
            return optim.SGD(
                params,
                lr=cfg.lr,
                momentum=cfg.momentum,
                weight_decay=cfg.weight_decay,
                nesterov=True,
            )
        elif name == "rmsprop":
            return optim.RMSprop(
                params, lr=cfg.lr, weight_decay=cfg.weight_decay
            )
        raise ValueError(f"Unknown optimizer: '{cfg.name}'")

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying model, unwrapping DataParallel if present."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module
        return self.model

    def _resume(self, path: Path) -> int:
        """
        Load training state from a checkpoint.

        Returns:
            The epoch to resume from (last saved epoch + 1).
        """
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        ckpt: dict = torch.load(path, map_location="cpu", weights_only=False)
        self._unwrap_model().load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if "scaler_state_dict" in ckpt and ckpt["scaler_state_dict"] and self.scaler:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])

        if "early_stopping_state" in ckpt and ckpt["early_stopping_state"]:
            self.early_stopper.load_state_dict(ckpt["early_stopping_state"])

        epoch = ckpt.get("epoch", 0)
        logger.info("Resumed from %s at epoch %d.", path.name, epoch)
        return epoch + 1

    def _log_epoch(
        self, epoch: int, metrics: dict[str, Any]
    ) -> None:
        """Write one row to the CSV log file."""
        if self._csv_logger is None:
            self._csv_logger = _CSVLogger(
                self.config.log_dir / self.config.experiment_name / "training_log.csv",
                fieldnames=list(metrics.keys()),
            )
        self._csv_logger.log(metrics)

    def _log_to_tb(self, epoch: int, metrics: dict[str, Any]) -> None:
        """Write scalars to TensorBoard."""
        if self._tb_writer is None:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._tb_writer.add_scalar(k, v, global_step=epoch)
