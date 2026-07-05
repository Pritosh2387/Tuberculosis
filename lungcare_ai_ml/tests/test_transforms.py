"""
Tests for the Albumentations-based transform pipeline (transforms.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture()
def synthetic_image() -> np.ndarray:
    """Return a random 256×256×3 uint8 RGB image."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture()
def synthetic_mask() -> np.ndarray:
    """Return a random 256×256 binary uint8 mask."""
    return np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255


class TestTransforms:
    def test_train_transform_returns_tensor(self, synthetic_image: np.ndarray) -> None:
        from datasets.transforms import build_transforms

        transform = build_transforms(split="train", image_size=(224, 224))
        result = transform(image=synthetic_image)
        img = result["image"]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)

    def test_val_transform_returns_tensor(self, synthetic_image: np.ndarray) -> None:
        from datasets.transforms import build_transforms

        transform = build_transforms(split="val", image_size=(224, 224))
        result = transform(image=synthetic_image)
        img = result["image"]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)

    def test_mask_preserved_through_transform(
        self,
        synthetic_image: np.ndarray,
        synthetic_mask: np.ndarray,
    ) -> None:
        from datasets.transforms import build_transforms

        transform = build_transforms(split="val", image_size=(128, 128))
        result = transform(image=synthetic_image, mask=synthetic_mask)
        assert "mask" in result
        mask = result["mask"]
        # Mask should be resized to (128, 128)
        if isinstance(mask, torch.Tensor):
            assert mask.shape[-2:] == (128, 128)
        else:
            assert mask.shape[:2] == (128, 128)

    def test_different_sizes(self, synthetic_image: np.ndarray) -> None:
        from datasets.transforms import build_transforms

        for size in [(64, 64), (128, 256), (512, 512)]:
            transform = build_transforms(split="val", image_size=size)
            result = transform(image=synthetic_image)
            img = result["image"]
            assert img.shape[1:] == size

    def test_normalisation_range(self, synthetic_image: np.ndarray) -> None:
        """After normalisation, pixel values should not all be in [0, 1]."""
        from datasets.transforms import build_transforms

        transform = build_transforms(split="val", image_size=(224, 224))
        result = transform(image=synthetic_image)
        img = result["image"]
        # ImageNet norm shifts values outside [0, 1]
        assert img.min().item() < 0 or img.max().item() > 1

    def test_train_vs_val_determinism(self, synthetic_image: np.ndarray) -> None:
        """Val transforms should be deterministic; train transforms may not be."""
        from datasets.transforms import build_transforms

        val_transform = build_transforms(split="val", image_size=(224, 224))
        out1 = val_transform(image=synthetic_image)["image"]
        out2 = val_transform(image=synthetic_image)["image"]
        assert torch.allclose(out1, out2), "Val transforms must be deterministic"
