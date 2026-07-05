"""
Tests for metric classes (training/metrics.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestClassificationMetrics:
    def test_binary_update_compute(self) -> None:
        from training.metrics import ClassificationMetrics

        m = ClassificationMetrics(num_classes=1, task="binary")
        preds = torch.tensor([0.9, 0.1, 0.8, 0.2])
        targets = torch.tensor([1, 0, 1, 0])
        m.update(preds, targets)
        result = m.compute()
        assert "accuracy" in result
        assert "f1" in result
        assert "auroc" in result
        assert result["accuracy"] == pytest.approx(1.0, abs=1e-3)

    def test_multiclass_keys(self) -> None:
        from training.metrics import ClassificationMetrics

        m = ClassificationMetrics(num_classes=3, task="multiclass")
        # Uniform prediction
        preds = torch.softmax(torch.randn(10, 3), dim=1)
        targets = torch.randint(0, 3, (10,))
        m.update(preds, targets)
        result = m.compute()
        assert "accuracy" in result
        assert "f1_macro" in result
        assert "f1_weighted" in result
        assert "auroc" in result

    def test_multilabel_keys(self) -> None:
        from training.metrics import ClassificationMetrics

        m = ClassificationMetrics(num_classes=4, task="multilabel")
        preds = torch.sigmoid(torch.randn(8, 4))
        targets = torch.randint(0, 2, (8, 4))
        m.update(preds, targets)
        result = m.compute()
        assert "accuracy" in result
        assert "f1_macro" in result

    def test_reset_clears_state(self) -> None:
        from training.metrics import ClassificationMetrics

        m = ClassificationMetrics(num_classes=1, task="binary")
        preds = torch.tensor([0.9, 0.1])
        targets = torch.tensor([1, 0])
        m.update(preds, targets)
        m.reset()
        # After reset, update+compute should only reflect new data
        m.update(torch.tensor([0.05]), torch.tensor([0]))
        result = m.compute()
        assert result["accuracy"] == pytest.approx(1.0, abs=1e-3)

    def test_accumulation_across_batches(self) -> None:
        from training.metrics import ClassificationMetrics

        m = ClassificationMetrics(num_classes=2, task="multiclass")
        for _ in range(5):
            preds = torch.softmax(torch.randn(4, 2), dim=1)
            targets = torch.randint(0, 2, (4,))
            m.update(preds, targets)
        result = m.compute()
        assert 0.0 <= result["accuracy"] <= 1.0

    @pytest.mark.parametrize("class_names", [
        None,
        ["Healthy", "TB", "Pneumonia", "COVID-19", "Cancer", "Fibrosis"],
    ])
    def test_per_class_f1_in_multiclass(self, class_names) -> None:
        from training.metrics import ClassificationMetrics

        m = ClassificationMetrics(
            num_classes=6, task="multiclass", class_names=class_names
        )
        preds = torch.softmax(torch.randn(16, 6), dim=1)
        targets = torch.randint(0, 6, (16,))
        m.update(preds, targets)
        result = m.compute()
        # At least one per-class F1 key should exist
        f1_keys = [k for k in result if k.startswith("f1_") and k not in ("f1_macro", "f1_weighted")]
        assert len(f1_keys) > 0


class TestSegmentationMetrics:
    def test_perfect_binary_dice(self) -> None:
        from training.metrics import SegmentationMetrics

        m = SegmentationMetrics(num_classes=1, threshold=0.5)
        # Perfect prediction: positive logit where mask=1
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        logits = mask * 20.0 - (1 - mask) * 20.0
        m.update(logits, mask)
        result = m.compute()
        assert result["dice"] > 0.95

    def test_worst_binary_dice(self) -> None:
        from training.metrics import SegmentationMetrics

        m = SegmentationMetrics(num_classes=1, threshold=0.5)
        mask = torch.zeros(2, 1, 64, 64)
        mask[:, :, 16:48, 16:48] = 1.0
        logits = (1 - mask) * 20.0 - mask * 20.0
        m.update(logits, mask)
        result = m.compute()
        assert result["dice"] < 0.2

    def test_all_zero_mask_iou(self) -> None:
        from training.metrics import SegmentationMetrics

        m = SegmentationMetrics(num_classes=1, threshold=0.5)
        mask = torch.zeros(2, 1, 64, 64)
        logits = torch.full((2, 1, 64, 64), -10.0)
        m.update(logits, mask)
        result = m.compute()
        # All-zero prediction on all-zero mask → perfect pixel accuracy
        assert result["pixel_accuracy"] > 0.9

    def test_reset_behaviour(self) -> None:
        from training.metrics import SegmentationMetrics

        m = SegmentationMetrics(num_classes=1)
        mask = torch.zeros(1, 1, 64, 64)
        m.update(torch.full((1, 1, 64, 64), 10.0), mask)
        m.reset()
        mask2 = torch.ones(1, 1, 32, 32)
        logits2 = torch.full((1, 1, 32, 32), 10.0)
        m.update(logits2, mask2)
        result = m.compute()
        assert result["dice"] > 0.9
