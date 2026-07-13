"""
datasets/segmentation_dataset.py
──────────────────────────────────
CSV-driven lung segmentation dataset.

Returns::

    {
        "image":    FloatTensor (3, H, W),
        "label":    FloatTensor (1, H, W)  — binary mask {0.0, 1.0},
        "metadata": dict with image_path, mask_path, dataset
    }

CSV schema
----------
Required columns: ``image_path``, ``mask_path``
Optional columns: ``dataset``

Montgomery County dataset provides left+right lung masks.
``scripts/prepare_data.py`` merges them into one combined mask.

Interview note
--------------
Why does the mask go through the same transform as the image?
Geometric augmentations (flip, rotate) must be applied IDENTICALLY to
both image and mask. Albumentations handles this via
``additional_targets={'mask': 'mask'}`` in build_seg_transforms().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.transforms import build_seg_transforms

logger = logging.getLogger("lungcare.dataset.segmentation")


def _load_image_rgb(path: Path) -> np.ndarray:
    """Load image as (H, W, 3) uint8 RGB array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2.imread failed: {path}")
    if img.dtype != np.uint8:
        img = (img / max(img.max(), 1) * 255).clip(0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _load_mask_binary(path: Path) -> np.ndarray:
    """
    Load a mask PNG and return (H, W) uint8 with values {0, 1}.

    Thresholding at 127 forces truly binary values, cleaning up any
    JPEG-compression artefacts in the mask files.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"cv2.imread failed for mask: {path}")
    return (mask > 127).astype(np.uint8)


class SegmentationDataset(Dataset):
    """
    Lung segmentation dataset driven by a CSV manifest.

    Args:
        csv_path:   Path to the split CSV file.
        split:      ``'train'``, ``'val'``, or ``'test'``.
        image_size: Square resize target in pixels.
        transform:  Optional pre-built segmentation pipeline.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str = "train",
        image_size: int = 224,
        transform: Any = None,
    ) -> None:
        self.split      = split
        self.image_size = image_size
        self.transform  = transform or build_seg_transforms(split, image_size)

        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        missing = {"image_path", "mask_path"} - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        logger.info("SegmentationDataset | split=%s | samples=%d",
                    split, len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row        = self.df.iloc[idx]
        image_path = Path(str(row["image_path"]))
        mask_path  = Path(str(row["mask_path"]))

        try:
            image_np = _load_image_rgb(image_path)
            mask_np  = _load_mask_binary(mask_path)
        except Exception as exc:
            logger.warning("Failed to load sample %d: %s", idx, exc)
            image_np = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            mask_np  = np.zeros((self.image_size, self.image_size),    dtype=np.uint8)

        result       = self.transform(image=image_np, mask=mask_np)
        image_tensor = result["image"]                                 # (3, H, W)
        raw_mask     = result["mask"]
        if isinstance(raw_mask, torch.Tensor):
            mask_tensor = raw_mask.float().unsqueeze(0)                # (1, H, W)
        else:
            mask_tensor = torch.from_numpy(
                raw_mask.astype(np.float32)
            ).unsqueeze(0)                                             # (1, H, W)

        return {
            "image": image_tensor,
            "label": mask_tensor,
            "metadata": {
                "image_path": str(image_path),
                "mask_path":  str(mask_path),
                "dataset":    str(row.get("dataset", "")),
                "idx":        idx,
            },
        }
