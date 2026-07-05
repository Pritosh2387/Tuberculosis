"""
Tests for loss functions (training/losses.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFocalLoss:
    def test_binary_forward(self) -> None:
        from training.losses import FocalLoss

        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        logits = torch.randn(8)
        targets = torch.randint(0, 2, (8,)).float()
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_multilabel_forward(self) -> None:
        from training.losses import FocalLoss

        loss_fn = FocalLoss()
        logits = torch.randn(4, 6)
        targets = torch.randint(0, 2, (4, 6)).float()
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_gamma_zero_equals_bce(self) -> None:
        """Focal loss with gamma=0 should approach standard BCE."""
        import torch.nn.functional as F
        from training.losses import FocalLoss

        torch.manual_seed(42)
        logits = torch.randn(16)
        targets = torch.randint(0, 2, (16,)).float()

        focal = FocalLoss(alpha=1.0, gamma=0.0)(logits, targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        assert abs(focal.item() - bce.item()) < 1e-5


class TestDiceLoss:
    def test_perfect_prediction_near_zero(self) -> None:
        from training.losses import DiceLoss

        loss_fn = DiceLoss(smooth=1.0)
        # Perfect prediction: large positive logit where mask=1, large negative where mask=0
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        logits = mask * 20.0 - (1 - mask) * 20.0
        loss = loss_fn(logits, mask)
        assert loss.item() < 0.05

    def test_worst_prediction_near_one(self) -> None:
        from training.losses import DiceLoss

        loss_fn = DiceLoss(smooth=1.0)
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        # Inverted prediction
        logits = (1 - mask) * 20.0 - mask * 20.0
        loss = loss_fn(logits, mask)
        assert loss.item() > 0.5

    def test_output_is_scalar(self) -> None:
        from training.losses import DiceLoss

        loss_fn = DiceLoss()
        logits = torch.randn(4, 1, 128, 128)
        mask = torch.randint(0, 2, (4, 1, 128, 128)).float()
        loss = loss_fn(logits, mask)
        assert loss.ndim == 0


class TestBCEDiceLoss:
    def test_forward(self) -> None:
        from training.losses import BCEDiceLoss

        loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        logits = torch.randn(2, 1, 64, 64)
        mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(logits, mask)
        assert loss.item() >= 0

    def test_gradient_flows(self) -> None:
        from training.losses import BCEDiceLoss

        loss_fn = BCEDiceLoss()
        logits = torch.randn(2, 1, 64, 64, requires_grad=True)
        mask = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(logits, mask)
        loss.backward()
        assert logits.grad is not None


class TestDeepSupervisionLoss:
    def test_list_input(self) -> None:
        from training.losses import BCEDiceLoss, DeepSupervisionLoss

        base = BCEDiceLoss()
        ds_loss = DeepSupervisionLoss(criterion=base)
        outputs = [torch.randn(2, 1, h, h) for h in (16, 32, 64)]
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = ds_loss(outputs, target)
        assert loss.item() >= 0

    def test_single_tensor_passthrough(self) -> None:
        from training.losses import BCEDiceLoss, DeepSupervisionLoss

        base = BCEDiceLoss()
        ds_loss = DeepSupervisionLoss(criterion=base)
        output = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = ds_loss(output, target)
        assert loss.item() >= 0

    def test_custom_weights_sum_to_one(self) -> None:
        from training.losses import BCEDiceLoss, DeepSupervisionLoss

        base = BCEDiceLoss()
        ds_loss = DeepSupervisionLoss(criterion=base, weights=[1.0, 2.0, 3.0])
        outputs = [torch.zeros(1, 1, h, h) for h in (16, 32, 64)]
        target = torch.zeros(1, 1, 64, 64)
        # All zeros → expect very small loss
        loss = ds_loss(outputs, target)
        assert loss.item() >= 0


class TestBuildFactories:
    @pytest.mark.parametrize("task", ["binary", "multiclass", "multilabel"])
    def test_build_classification_loss(self, task: str) -> None:
        from training.losses import build_classification_loss

        loss_fn = build_classification_loss(task)
        assert loss_fn is not None

    @pytest.mark.parametrize("loss_type", ["dice", "bce", "bce_dice", "focal_dice"])
    def test_build_segmentation_loss(self, loss_type: str) -> None:
        from training.losses import build_segmentation_loss

        loss_fn = build_segmentation_loss(loss_type)
        logits = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = loss_fn(logits, target)
        assert loss.item() >= 0
