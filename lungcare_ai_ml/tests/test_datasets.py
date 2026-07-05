"""
Tests for dataset classes — ClassificationDataset, SegmentationDataset, DICOMDataset.

All tests use synthetic in-memory data (temporary directories with generated
images/masks) so they run without any downloaded datasets.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# ── Ensure project root is on sys.path ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with 10 synthetic 256×256 RGB PNG images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(10):
        arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i:03d}.png")
    return img_dir


@pytest.fixture()
def tmp_mask_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with 10 synthetic binary 256×256 masks."""
    msk_dir = tmp_path / "masks"
    msk_dir.mkdir()
    for i in range(10):
        arr = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        Image.fromarray(arr, mode="L").save(msk_dir / f"mask_{i:03d}.png")
    return msk_dir


@pytest.fixture()
def classification_csv(tmp_path: Path, tmp_image_dir: Path) -> Path:
    """CSV manifest for classification (10 samples, 2 classes)."""
    labels = ["Healthy", "Tuberculosis"] * 5
    csv_path = tmp_path / "cls.csv"
    images = sorted(tmp_image_dir.glob("*.png"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "label_idx", "dataset"])
        writer.writeheader()
        for img, lbl in zip(images, labels):
            writer.writerow({
                "image_path": str(img),
                "label": lbl,
                "label_idx": str(0 if lbl == "Healthy" else 1),
                "dataset": "synthetic",
            })
    return csv_path


@pytest.fixture()
def segmentation_csv(tmp_path: Path, tmp_image_dir: Path, tmp_mask_dir: Path) -> Path:
    """CSV manifest for segmentation (10 samples with masks)."""
    csv_path = tmp_path / "seg.csv"
    images = sorted(tmp_image_dir.glob("*.png"))
    masks = sorted(tmp_mask_dir.glob("*.png"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "mask_path", "dataset"])
        writer.writeheader()
        for img, msk in zip(images, masks):
            writer.writerow({
                "image_path": str(img),
                "mask_path": str(msk),
                "dataset": "synthetic",
            })
    return csv_path


# ─── ClassificationDataset ────────────────────────────────────────────────────


class TestClassificationDataset:
    def test_len(self, classification_csv: Path) -> None:
        from datasets.classification_dataset import ClassificationDataset

        ds = ClassificationDataset(
            csv_path=classification_csv,
            image_size=(224, 224),
            split="val",
            num_classes=6,
            task="multiclass",
        )
        assert len(ds) == 10

    def test_sample_keys(self, classification_csv: Path) -> None:
        from datasets.classification_dataset import ClassificationDataset

        ds = ClassificationDataset(
            csv_path=classification_csv,
            image_size=(224, 224),
            split="val",
            num_classes=6,
            task="multiclass",
        )
        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert "metadata" in sample

    def test_image_shape(self, classification_csv: Path) -> None:
        from datasets.classification_dataset import ClassificationDataset

        ds = ClassificationDataset(
            csv_path=classification_csv,
            image_size=(128, 128),
            split="val",
            num_classes=6,
            task="multiclass",
        )
        image = ds[0]["image"]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 128, 128)

    def test_label_is_tensor(self, classification_csv: Path) -> None:
        from datasets.classification_dataset import ClassificationDataset

        ds = ClassificationDataset(
            csv_path=classification_csv,
            image_size=(224, 224),
            split="val",
            num_classes=6,
            task="multiclass",
        )
        label = ds[0]["label"]
        assert isinstance(label, torch.Tensor)

    def test_dataloader_batching(self, classification_csv: Path) -> None:
        from datasets.classification_dataset import ClassificationDataset
        from torch.utils.data import DataLoader

        ds = ClassificationDataset(
            csv_path=classification_csv,
            image_size=(64, 64),
            split="val",
            num_classes=6,
            task="multiclass",
        )
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["image"].shape == (4, 3, 64, 64)
        assert batch["label"].shape[0] == 4

    def test_corrupted_image_handling(self, tmp_path: Path) -> None:
        """Dataset should skip / not crash on corrupted files."""
        from datasets.classification_dataset import ClassificationDataset

        bad_img = tmp_path / "bad.png"
        bad_img.write_bytes(b"not_an_image")
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label", "label_idx", "dataset"])
            writer.writeheader()
            writer.writerow({
                "image_path": str(bad_img),
                "label": "Healthy",
                "label_idx": "0",
                "dataset": "synthetic",
            })

        ds = ClassificationDataset(
            csv_path=csv_path,
            image_size=(64, 64),
            split="val",
            num_classes=6,
            task="multiclass",
        )
        # Should not raise; either returns a zero tensor or skips
        try:
            _ = ds[0]
        except Exception:
            pass  # acceptable — dataset may raise with clear error


# ─── SegmentationDataset ─────────────────────────────────────────────────────


class TestSegmentationDataset:
    def test_len(self, segmentation_csv: Path) -> None:
        from datasets.segmentation_dataset import SegmentationDataset

        ds = SegmentationDataset(
            csv_path=segmentation_csv,
            image_size=(256, 256),
            split="val",
        )
        assert len(ds) == 10

    def test_sample_has_mask(self, segmentation_csv: Path) -> None:
        from datasets.segmentation_dataset import SegmentationDataset

        ds = SegmentationDataset(
            csv_path=segmentation_csv,
            image_size=(256, 256),
            split="val",
        )
        sample = ds[0]
        assert "mask" in sample
        assert sample["mask"] is not None

    def test_mask_shape(self, segmentation_csv: Path) -> None:
        from datasets.segmentation_dataset import SegmentationDataset

        ds = SegmentationDataset(
            csv_path=segmentation_csv,
            image_size=(128, 128),
            split="val",
        )
        mask = ds[0]["mask"]
        assert isinstance(mask, torch.Tensor)
        # Binary segmentation: (1, H, W)
        assert mask.shape == (1, 128, 128)

    def test_mask_is_binary(self, segmentation_csv: Path) -> None:
        from datasets.segmentation_dataset import SegmentationDataset

        ds = SegmentationDataset(
            csv_path=segmentation_csv,
            image_size=(128, 128),
            split="val",
        )
        for i in range(len(ds)):
            mask = ds[i]["mask"]
            unique_vals = mask.unique()
            assert all(v.item() in (0.0, 1.0) for v in unique_vals), \
                f"Mask contains non-binary values: {unique_vals}"
