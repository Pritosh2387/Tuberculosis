"""
Integration tests for LungCare AI — end-to-end mini training + inference runs.

These tests exercise the full stack:
  Dataset → DataLoader → Model → Trainer (2 epochs) → Evaluator → Pipeline

All tests use:
  - Synthetic in-memory data (no real datasets required)
  - Tiny model configurations for fast CPU execution
  - Minimal epochs (2) to verify loop correctness, not convergence
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_cls_data(tmp_path_factory: pytest.TempPathFactory):
    """
    Build a tiny classification dataset (20 images, 2 classes).
    Returns (tmp_dir, train_csv, val_csv).
    """
    tmp = tmp_path_factory.mktemp("cls_data")
    img_dir = tmp / "images"
    img_dir.mkdir()

    labels = ["Healthy"] * 10 + ["Tuberculosis"] * 10
    class_to_idx = {"Healthy": 0, "Tuberculosis": 1}
    rows = []

    for i, lbl in enumerate(labels):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        p = img_dir / f"img_{i:03d}.png"
        Image.fromarray(arr).save(p)
        rows.append({"image_path": str(p), "label": lbl,
                     "label_idx": str(class_to_idx[lbl]), "dataset": "synthetic"})

    def write(path, data):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    train_csv = tmp / "train.csv"
    val_csv = tmp / "val.csv"
    write(train_csv, rows[:14])
    write(val_csv, rows[14:])
    return tmp, train_csv, val_csv


@pytest.fixture(scope="module")
def synthetic_seg_data(tmp_path_factory: pytest.TempPathFactory):
    """Build a tiny segmentation dataset (10 image/mask pairs)."""
    tmp = tmp_path_factory.mktemp("seg_data")
    img_dir = tmp / "images"
    msk_dir = tmp / "masks"
    img_dir.mkdir()
    msk_dir.mkdir()

    rows = []
    for i in range(10):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        msk = (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255
        ip = img_dir / f"img_{i}.png"
        mp = msk_dir / f"msk_{i}.png"
        Image.fromarray(arr).save(ip)
        Image.fromarray(msk, mode="L").save(mp)
        rows.append({"image_path": str(ip), "mask_path": str(mp), "dataset": "synthetic"})

    def write(path, data):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    train_csv = tmp / "train.csv"
    val_csv = tmp / "val.csv"
    write(train_csv, rows[:7])
    write(val_csv, rows[7:])
    return tmp, train_csv, val_csv


# ─── Classification integration ───────────────────────────────────────────────


class TestClassificationIntegration:
    def test_full_training_loop(
        self, synthetic_cls_data, tmp_path: Path
    ) -> None:
        from torch.utils.data import DataLoader
        from datasets.classification_dataset import ClassificationDataset
        from models.classification.resnet import ResNet50Classifier
        from training.classification_trainer import ClassificationConfig, ClassificationTrainer
        from training.trainer import EarlyStoppingConfig, OptimizerConfig, SchedulerConfig

        _, train_csv, val_csv = synthetic_cls_data

        train_ds = ClassificationDataset(
            csv_path=train_csv, image_size=(64, 64),
            split="train", num_classes=2, task="multiclass",
            class_names=["Healthy", "Tuberculosis"],
        )
        val_ds = ClassificationDataset(
            csv_path=val_csv, image_size=(64, 64),
            split="val", num_classes=2, task="multiclass",
            class_names=["Healthy", "Tuberculosis"],
        )
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        model = ResNet50Classifier(num_classes=2, pretrained=False)

        cfg = ClassificationConfig(
            task="multiclass", num_classes=2, epochs=2,
            amp=False, grad_clip=1.0, grad_accum_steps=1,
            data_parallel=False,
            log_dir=tmp_path / "logs",
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_name="test_cls",
            monitor_metric="val_f1_macro",
            monitor_mode="max",
            optimizer=OptimizerConfig(name="adamw", lr=1e-3),
            scheduler=SchedulerConfig(name="cosine", T_max=2),
            early_stopping=EarlyStoppingConfig(enabled=False),
        )

        trainer = ClassificationTrainer(
            model=model, train_loader=train_loader,
            val_loader=val_loader, config=cfg, device="cpu",
            class_names=["Healthy", "Tuberculosis"],
        )
        trainer.train()  # Should complete without error

        # Checkpoint should be saved
        ckpt_dir = tmp_path / "checkpoints" / "test_cls"
        assert any(ckpt_dir.glob("*.pth"))

    def test_evaluator_on_synthetic(
        self, synthetic_cls_data, tmp_path: Path
    ) -> None:
        from torch.utils.data import DataLoader
        from datasets.classification_dataset import ClassificationDataset
        from models.classification.densenet import DenseNet121Classifier
        from evaluation.evaluator import Evaluator

        _, _, val_csv = synthetic_cls_data

        val_ds = ClassificationDataset(
            csv_path=val_csv, image_size=(64, 64),
            split="val", num_classes=2, task="multiclass",
        )
        val_loader = DataLoader(val_ds, batch_size=4)
        model = DenseNet121Classifier(num_classes=2, pretrained=False)

        evaluator = Evaluator(
            model=model, loader=val_loader,
            task="multiclass", num_classes=2, device="cpu",
        )
        result = evaluator.evaluate()
        assert "accuracy" in result.metrics
        assert 0.0 <= result.metrics["accuracy"] <= 1.0


# ─── Segmentation integration ─────────────────────────────────────────────────


class TestSegmentationIntegration:
    def test_unet_training_loop(
        self, synthetic_seg_data, tmp_path: Path
    ) -> None:
        from torch.utils.data import DataLoader
        from datasets.segmentation_dataset import SegmentationDataset
        from models.segmentation.unet import UNet
        from training.segmentation_trainer import SegmentationConfig, SegmentationTrainer
        from training.trainer import EarlyStoppingConfig, OptimizerConfig, SchedulerConfig

        _, train_csv, val_csv = synthetic_seg_data

        train_ds = SegmentationDataset(
            csv_path=train_csv, image_size=(64, 64), split="train", in_channels=3,
        )
        val_ds = SegmentationDataset(
            csv_path=val_csv, image_size=(64, 64), split="val", in_channels=3,
        )
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

        model = UNet(in_channels=3, out_channels=1, features=(8, 16, 32))

        cfg = SegmentationConfig(
            epochs=2, amp=False, grad_clip=1.0,
            log_dir=tmp_path / "logs",
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_name="test_seg",
            deep_supervision=False,
            optimizer=OptimizerConfig(name="adamw", lr=1e-3),
            scheduler=SchedulerConfig(name="cosine", T_max=2),
            early_stopping=EarlyStoppingConfig(enabled=False),
        )

        trainer = SegmentationTrainer(
            model=model, train_loader=train_loader,
            val_loader=val_loader, config=cfg, device="cpu",
        )
        trainer.train()

    def test_unetpp_deep_supervision_training(
        self, synthetic_seg_data, tmp_path: Path
    ) -> None:
        from torch.utils.data import DataLoader
        from datasets.segmentation_dataset import SegmentationDataset
        from models.segmentation.unet_plus_plus import UNetPlusPlus
        from training.segmentation_trainer import SegmentationConfig, SegmentationTrainer
        from training.trainer import EarlyStoppingConfig, OptimizerConfig, SchedulerConfig

        _, train_csv, val_csv = synthetic_seg_data

        train_ds = SegmentationDataset(
            csv_path=train_csv, image_size=(64, 64), split="train", in_channels=3,
        )
        val_ds = SegmentationDataset(
            csv_path=val_csv, image_size=(64, 64), split="val", in_channels=3,
        )
        train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=2)

        model = UNetPlusPlus(
            in_channels=3, out_channels=1,
            features=(8, 16, 32), deep_supervision=True
        )

        cfg = SegmentationConfig(
            epochs=2, amp=False, grad_clip=1.0,
            log_dir=tmp_path / "logs",
            checkpoint_dir=tmp_path / "checkpoints",
            experiment_name="test_unetpp",
            deep_supervision=True,
            ds_weights=[0.3, 0.7],
            optimizer=OptimizerConfig(name="adamw", lr=1e-3),
            scheduler=SchedulerConfig(name="cosine", T_max=2),
            early_stopping=EarlyStoppingConfig(enabled=False),
        )

        trainer = SegmentationTrainer(
            model=model, train_loader=train_loader,
            val_loader=val_loader, config=cfg, device="cpu",
        )
        trainer.train()

    def test_report_generator(self) -> None:
        from evaluation.report_generator import ReportGenerator

        class_names = ["Healthy", "Tuberculosis", "Pneumonia",
                       "COVID-19", "Lung Cancer", "Pulmonary Fibrosis"]
        gen = ReportGenerator(class_names=class_names)
        probs = np.array([0.03, 0.92, 0.02, 0.01, 0.01, 0.01], dtype=np.float32)
        report = gen.generate(probs=probs, case_id="integration_test")

        assert report["status"] == "abnormal"
        assert report["prediction"] == "Tuberculosis"
        assert report["confidence"] == pytest.approx(0.92, abs=1e-3)
        assert len(report["findings"]) > 0

        md = gen.to_markdown(report)
        assert "Tuberculosis" in md
        assert "## Findings" in md
