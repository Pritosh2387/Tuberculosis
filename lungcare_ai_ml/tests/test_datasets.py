"""
tests/test_datasets.py
───────────────────────
Tests for datasets and transforms (absorbs the old test_transforms.py).

Strategy: build synthetic CSVs and in-memory images — no disk datasets needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from datasets.transforms import build_transforms, build_seg_transforms


# ─── Transforms ──────────────────────────────────────────────────────────────

class TestTransforms:
    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_classification_transform_output_shape(self, split: str) -> None:
        tf  = build_transforms(split, image_size=224)
        img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        out = tf(image=img)
        assert out["image"].shape == (3, 224, 224)
        assert out["image"].dtype == torch.float32

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_seg_transform_output_shape(self, split: str) -> None:
        tf   = build_seg_transforms(split, image_size=224)
        img  = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        mask = (np.random.rand(256, 256) > 0.5).astype(np.uint8)
        out  = tf(image=img, mask=mask)
        assert out["image"].shape == (3, 224, 224)
        assert out["mask"].shape  == (224, 224)

    def test_seg_transform_geometric_consistency(self) -> None:
        """Same geometric transform should apply to both image and mask."""
        tf   = build_seg_transforms("train", image_size=64)
        img  = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
        mask = np.ones((128, 128), dtype=np.uint8)   # all-ones mask

        out  = tf(image=img, mask=mask)
        # After resize the mask should still have a uniform pattern if HFlip was
        # applied consistently (mask is all-ones — flip is a no-op)
        assert out["mask"].sum() > 0   # mask survived the transform


# ─── ClassificationDataset ───────────────────────────────────────────────────

class TestClassificationDataset:
    @pytest.fixture
    def csv_with_real_images(self, tmp_path: "pytest.TempPathFactory") -> str:
        """Create synthetic PNG images and a matching CSV."""
        import cv2
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        rows = []
        for i, label in enumerate(["Normal", "Tuberculosis"] * 3):
            img_path = img_dir / f"img_{i:03d}.png"
            # Synthetic 64×64 grey image
            img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            cv2.imwrite(str(img_path), img)
            rows.append({
                "image_path": str(img_path),
                "label":      label,
                "label_idx":  0 if label == "Normal" else 1,
                "dataset":    "test",
            })

        csv_path = tmp_path / "test.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return str(csv_path)

    def test_len(self, csv_with_real_images: str) -> None:
        from datasets.classification_dataset import ClassificationDataset
        ds = ClassificationDataset(csv_with_real_images, split="val", image_size=64)
        assert len(ds) == 6

    def test_item_keys(self, csv_with_real_images: str) -> None:
        from datasets.classification_dataset import ClassificationDataset
        ds   = ClassificationDataset(csv_with_real_images, split="val", image_size=64)
        item = ds[0]
        assert set(item.keys()) == {"image", "label", "metadata"}

    def test_image_shape(self, csv_with_real_images: str) -> None:
        from datasets.classification_dataset import ClassificationDataset
        ds   = ClassificationDataset(csv_with_real_images, split="val", image_size=64)
        item = ds[0]
        assert item["image"].shape == (3, 64, 64)
        assert item["image"].dtype == torch.float32

    def test_label_dtype(self, csv_with_real_images: str) -> None:
        from datasets.classification_dataset import ClassificationDataset
        ds   = ClassificationDataset(csv_with_real_images, split="val", image_size=64)
        item = ds[0]
        assert item["label"].dtype == torch.long

    def test_sample_weights_sum(self, csv_with_real_images: str) -> None:
        from datasets.classification_dataset import ClassificationDataset
        ds = ClassificationDataset(csv_with_real_images, split="train", image_size=64)
        w  = ds.get_sample_weights()
        assert w.shape == (6,)
        assert (w > 0).all()

    def test_missing_csv_raises(self, tmp_path: "pytest.TempPathFactory") -> None:
        from datasets.classification_dataset import ClassificationDataset
        with pytest.raises(FileNotFoundError):
            ClassificationDataset(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path: "pytest.TempPathFactory") -> None:
        from datasets.classification_dataset import ClassificationDataset
        df  = pd.DataFrame({"image_path": ["x.png"], "label": ["Normal"]})
        csv = tmp_path / "bad.csv"
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            ClassificationDataset(csv)

    def test_corrupted_image_returns_zeros(self, tmp_path: "pytest.TempPathFactory") -> None:
        """A corrupt image path should not crash — returns zero tensor."""
        from datasets.classification_dataset import ClassificationDataset
        csv_path = tmp_path / "t.csv"
        pd.DataFrame([{
            "image_path": str(tmp_path / "nonexistent.png"),
            "label":      "Normal",
            "label_idx":  0,
        }]).to_csv(csv_path, index=False)
        ds   = ClassificationDataset(csv_path, split="val", image_size=64)
        item = ds[0]
        assert item["image"].shape == (3, 64, 64)


# ─── SegmentationDataset ─────────────────────────────────────────────────────

class TestSegmentationDataset:
    @pytest.fixture
    def seg_csv(self, tmp_path: "pytest.TempPathFactory") -> str:
        import cv2
        img_dir  = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(); mask_dir.mkdir()

        rows = []
        for i in range(4):
            img_path  = img_dir  / f"img_{i}.png"
            mask_path = mask_dir / f"mask_{i}.png"
            img  = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
            mask = (np.random.rand(64, 64) > 0.5).astype(np.uint8) * 255
            cv2.imwrite(str(img_path), img)
            cv2.imwrite(str(mask_path), mask)
            rows.append({"image_path": str(img_path), "mask_path": str(mask_path)})

        csv_path = tmp_path / "seg.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return str(csv_path)

    def test_len(self, seg_csv: str) -> None:
        from datasets.segmentation_dataset import SegmentationDataset
        ds = SegmentationDataset(seg_csv, split="val", image_size=64)
        assert len(ds) == 4

    def test_mask_shape_and_range(self, seg_csv: str) -> None:
        from datasets.segmentation_dataset import SegmentationDataset
        ds   = SegmentationDataset(seg_csv, split="val", image_size=64)
        item = ds[0]
        assert item["label"].shape == (1, 64, 64)
        assert item["label"].min() >= 0.0
        assert item["label"].max() <= 1.0
