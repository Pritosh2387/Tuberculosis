"""
Segmentation trainer for LungCare AI.

Extends :class:`BaseTrainer` with:
- Binary / multiclass segmentation loss selection.
- U-Net++ deep supervision: handles both list outputs (training) and
  single tensor outputs (inference) transparently.
- :class:`SegmentationMetrics` (Dice, IoU, pixel accuracy) per epoch.
- Mask visualisation to TensorBoard every N epochs.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.losses import build_segmentation_loss
from training.metrics import SegmentationMetrics
from training.trainer import BaseTrainer, TrainerConfig

logger = logging.getLogger("lungcare.training.segmentation")


class SegmentationConfig(TrainerConfig):
    """
    Extended :class:`TrainerConfig` for segmentation training.

    Extra fields
    ------------
    loss_type: ``'dice'``, ``'bce'``, ``'bce_dice'``, or ``'focal_dice'``.
    bce_weight: BCE term weight in combined losses.
    dice_weight: Dice term weight in combined losses.
    dice_smooth: Dice loss smoothing constant.
    deep_supervision: Whether model uses U-Net++ deep supervision.
    ds_weights: Per-level weights for deep supervision (coarse → fine).
        Defaults to uniform weighting across output levels.
    mask_log_interval: Log mask visualisation to TensorBoard every N epochs.
    threshold: Sigmoid threshold for binary mask prediction (metrics).
    """

    loss_type: str = "bce_dice"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    dice_smooth: float = 1.0
    deep_supervision: bool = False
    ds_weights: list[float] | None = None
    mask_log_interval: int = 10
    threshold: float = 0.5
    monitor_metric: str = "val_dice"
    monitor_mode: str = "max"


class SegmentationTrainer(BaseTrainer):
    """
    Segmentation trainer supporting U-Net, Attention U-Net, and U-Net++.

    Handles U-Net++ deep supervision transparently: when the model is in
    ``model.training`` mode, its ``forward()`` returns a list of logit maps
    (coarse → fine) and :class:`DeepSupervisionLoss` computes a weighted
    sum.  In eval mode, a single tensor is returned and the base loss is
    applied directly.

    Args:
        model: Segmentation model (U-Net / Attention U-Net / U-Net++).
        train_loader: Training DataLoader (returns dicts with 'image' and 'mask').
        val_loader: Validation DataLoader.
        config: :class:`SegmentationConfig` instance.
        device: Target device.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: SegmentationConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(model, train_loader, val_loader, config, device)
        self.seg_config = config

        # ── Loss ──────────────────────────────────────────────────────────────
        self.criterion = build_segmentation_loss(
            loss_type=config.loss_type,
            bce_weight=config.bce_weight,
            dice_weight=config.dice_weight,
            smooth=config.dice_smooth,
            deep_supervision=config.deep_supervision,
            ds_weights=config.ds_weights,
        )

        # ── Metrics ────────────────────────────────────────────────────────────
        seg_classes = max(config.num_classes, 1)
        self._train_metrics = SegmentationMetrics(
            num_classes=seg_classes,
            threshold=config.threshold,
            device=self.device,
        )
        self._val_metrics = SegmentationMetrics(
            num_classes=seg_classes,
            threshold=config.threshold,
            device=self.device,
        )

        self._last_val_metrics: dict[str, float] = {}

        logger.info(
            "SegmentationTrainer | loss=%s | deep_supervision=%s | "
            "threshold=%.2f",
            config.loss_type,
            config.deep_supervision,
            config.threshold,
        )

    # ─── Step implementations ─────────────────────────────────────────────────

    def _train_step(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        output = self.model(images)

        # U-Net++ returns list during training when deep_supervision=True
        if isinstance(output, list):
            loss = self.criterion(output, masks)
            pred_logit = output[-1]    # Use the finest (last) output for metrics
        else:
            loss = self.criterion(output, masks)
            pred_logit = output

        self._train_metrics.update(pred_logit.detach(), masks)
        return loss

    @torch.no_grad()
    def _val_step(self, batch: dict[str, Any]) -> torch.Tensor:
        images = batch["image"].to(self.device)
        masks = batch["mask"].to(self.device)

        output = self.model(images)

        # Eval mode always returns a single tensor
        if isinstance(output, list):
            output = output[-1]

        # For validation, use the base loss (no deep supervision)
        if self.seg_config.deep_supervision:
            from training.losses import build_segmentation_loss
            base_loss = build_segmentation_loss(
                loss_type=self.seg_config.loss_type,
                bce_weight=self.seg_config.bce_weight,
                dice_weight=self.seg_config.dice_weight,
                smooth=self.seg_config.dice_smooth,
                deep_supervision=False,
            )
            loss = base_loss(output, masks)
        else:
            loss = self.criterion(output, masks)

        self._val_metrics.update(output, masks)
        return loss

    # ─── Epoch hooks ─────────────────────────────────────────────────────────

    def _pre_train_epoch(self, epoch: int) -> None:
        self._train_metrics.reset()

    def _compute_train_metrics(self) -> dict[str, float]:
        return self._train_metrics.compute()

    def _compute_val_metrics(self) -> dict[str, float]:
        self._last_val_metrics = self._val_metrics.compute()
        self._val_metrics.reset()
        return self._last_val_metrics

    def _get_monitor_value(self) -> float:
        key = self.config.monitor_metric.removeprefix("val_")
        return self._last_val_metrics.get(key, 0.0)

    # ─── TensorBoard mask visualisation ──────────────────────────────────────

    def log_mask_grid(self, epoch: int) -> None:
        """
        Log a grid of [image | prediction | ground-truth] to TensorBoard.

        Pulls the first batch from the validation loader.  Safe to call
        at the end of any epoch when a TensorBoard writer is available.
        """
        if self._tb_writer is None:
            return
        if epoch % self.seg_config.mask_log_interval != 0:
            return

        try:
            import torchvision.utils as vutils

            batch = next(iter(self.val_loader))
            images = batch["image"][:4].to(self.device)
            masks = batch["mask"][:4].to(self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(images)
                if isinstance(output, list):
                    output = output[-1]

            pred_masks = (torch.sigmoid(output[:4]) >= self.seg_config.threshold).float()
            # Normalise images for display
            imgs_disp = images[:, :1, :, :].expand(-1, 3, -1, -1)
            preds_disp = pred_masks.expand(-1, 3, -1, -1)
            gt_disp = masks[:, :1, :, :].expand(-1, 3, -1, -1)

            grid = vutils.make_grid(
                torch.cat([imgs_disp, preds_disp, gt_disp], dim=0),
                nrow=4, normalize=True,
            )
            self._tb_writer.add_image("Masks/img-pred-gt", grid, global_step=epoch)
        except Exception as exc:
            logger.warning("Mask grid logging failed: %s", exc)
