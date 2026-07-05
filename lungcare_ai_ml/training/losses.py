"""
Loss functions for LungCare AI training.

Classification losses
---------------------
- :class:`FocalLoss` — binary focal loss for class imbalance.
- :class:`LabelSmoothingCrossEntropy` — soft-target CE for multiclass.

Segmentation losses
-------------------
- :class:`DiceLoss` — differentiable Dice coefficient loss.
- :class:`BCEDiceLoss` — weighted BCE + Dice for segmentation.

Training utilities
------------------
- :class:`DeepSupervisionLoss` — wraps any criterion for U-Net++ multi-output.
- :func:`build_classification_loss` — factory from task string.
- :func:`build_segmentation_loss` — factory from loss type string.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("lungcare.training.losses")


# ─── Classification losses ────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for binary / multilabel classification.

    Reduces the relative loss of well-classified examples, focusing
    training on hard negatives.  Follows Lin et al. (2017).

    Args:
        alpha: Weighting factor for the positive class (0–1).
        gamma: Focusing exponent (≥ 0).  Higher → more focus on hard examples.
        reduction: ``'mean'``, ``'sum'``, or ``'none'``.
        pos_weight: Passed to :func:`F.binary_cross_entropy_with_logits`.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits ``(B, *)`` or ``(B, C, *)`` for multilabel.
            targets: Binary targets of the same shape as *inputs*.

        Returns:
            Scalar focal loss.
        """
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets.float(),
            pos_weight=self.pos_weight,
            reduction="none",
        )
        pt = torch.exp(-bce)
        focal = self.alpha * (1.0 - pt) ** self.gamma * bce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing for multiclass classification.

    Prevents overconfident softmax by mixing hard labels with a uniform
    distribution over classes.

    Args:
        smoothing: Label smoothing factor ε ∈ [0, 1).
        reduction: ``'mean'`` or ``'sum'``.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits ``(B, num_classes)``.
            targets: Class indices ``(B,)`` long tensor.
        """
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)

        # Hard target
        with torch.no_grad():
            smooth_target = torch.full_like(log_probs, self.smoothing / (num_classes - 1))
            smooth_target.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_target * log_probs).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ─── Segmentation losses ──────────────────────────────────────────────────────


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Differentiable approximation of the Dice coefficient:

    .. math::
        \\mathcal{L}_{\\text{Dice}} = 1 - \\frac{2 |P \\cap T| + \\epsilon}{|P| + |T| + \\epsilon}

    Args:
        smooth: Additive smoothing constant ε.
        sigmoid: If ``True``, applies sigmoid to *inputs* before computing.
    """

    def __init__(self, smooth: float = 1.0, sigmoid: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits or probabilities ``(B, 1, H, W)``.
            targets: Binary mask ``(B, 1, H, W)`` float or long.

        Returns:
            Scalar Dice loss in ``[0, 1]``.
        """
        probs = torch.sigmoid(inputs) if self.sigmoid else inputs
        targets_f = targets.float()

        intersection = (probs * targets_f).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_f.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss for segmentation.

    Args:
        bce_weight: Weight for the BCE term.
        dice_weight: Weight for the Dice term.
        smooth: Dice smoothing constant.
        pos_weight: Positive class weight for BCE.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits ``(B, 1, H, W)``.
            targets: Binary mask ``(B, 1, H, W)``.
        """
        bce_loss = self.bce(inputs, targets.float())
        dice_loss = self.dice(inputs, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class DeepSupervisionLoss(nn.Module):
    """
    Multi-output loss wrapper for U-Net++ deep supervision.

    During training, U-Net++ returns a list of segmentation maps at
    progressively finer resolutions.  This module resizes the target mask
    to each output's spatial size, computes the base loss, and returns
    a weighted sum.

    Args:
        criterion: Base loss module (e.g. :class:`BCEDiceLoss`).
        weights: Coefficients for each output level (coarsest → finest).
            Defaults to uniform weighting.  Will be normalised to sum 1.
    """

    def __init__(
        self,
        criterion: nn.Module,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self._weights = weights

    def forward(
        self,
        outputs: list[torch.Tensor] | torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            outputs: List of logit maps from the decoder (coarse → fine).
                If a single tensor is passed, delegates directly to the
                base criterion.
            target: Full-resolution binary mask ``(B, 1, H, W)``.

        Returns:
            Weighted sum of per-level losses.
        """
        if not isinstance(outputs, list):
            return self.criterion(outputs, target)

        n = len(outputs)
        raw_weights = self._weights if self._weights is not None else [1.0 / n] * n
        total = sum(raw_weights)
        weights = [w / total for w in raw_weights]

        loss = torch.tensor(0.0, device=target.device, requires_grad=True)
        for output, w in zip(outputs, weights):
            if output.shape[2:] != target.shape[2:]:
                target_r = F.interpolate(
                    target.float(), size=output.shape[2:], mode="nearest"
                )
            else:
                target_r = target
            loss = loss + w * self.criterion(output, target_r)

        return loss


# ─── Factories ────────────────────────────────────────────────────────────────


def build_classification_loss(
    task: str,
    label_smoothing: float = 0.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    use_focal: bool = False,
    pos_weight: torch.Tensor | None = None,
) -> nn.Module:
    """
    Return the appropriate classification loss for *task*.

    Args:
        task: ``'binary'``, ``'multiclass'``, or ``'multilabel'``.
        label_smoothing: Smoothing for multiclass CE (0 = hard labels).
        focal_alpha: Alpha for binary/multilabel focal loss.
        focal_gamma: Gamma for binary/multilabel focal loss.
        use_focal: Use :class:`FocalLoss` instead of BCE for binary/multilabel.
        pos_weight: Positive class weight tensor for binary/multilabel BCE.

    Returns:
        A :class:`nn.Module` loss.
    """
    if task == "multiclass":
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    if use_focal:
        return FocalLoss(
            alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight
        )
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def build_segmentation_loss(
    loss_type: str = "bce_dice",
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    smooth: float = 1.0,
    pos_weight: torch.Tensor | None = None,
    deep_supervision: bool = False,
    ds_weights: list[float] | None = None,
) -> nn.Module:
    """
    Build a segmentation loss, optionally wrapping with deep supervision.

    Args:
        loss_type: ``'dice'``, ``'bce'``, ``'bce_dice'``, or ``'focal_dice'``.
        bce_weight: BCE term weight in combined losses.
        dice_weight: Dice term weight in combined losses.
        smooth: Dice smoothing constant.
        pos_weight: Positive class weight for BCE.
        deep_supervision: If ``True``, wrap with :class:`DeepSupervisionLoss`.
        ds_weights: Per-level weights for deep supervision.

    Returns:
        A :class:`nn.Module` loss (possibly wrapped in DeepSupervisionLoss).
    """
    if loss_type == "dice":
        base: nn.Module = DiceLoss(smooth=smooth)
    elif loss_type == "bce":
        base = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal_dice":
        focal = FocalLoss(reduction="mean")
        dice = DiceLoss(smooth=smooth)

        class _FocalDice(nn.Module):
            def forward(self_, inp: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
                return 0.5 * focal(inp, tgt) + 0.5 * dice(inp, tgt)

        base = _FocalDice()
    else:
        base = BCEDiceLoss(
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            smooth=smooth,
            pos_weight=pos_weight,
        )

    if deep_supervision:
        return DeepSupervisionLoss(criterion=base, weights=ds_weights)
    return base
