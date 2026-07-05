"""
Segmentation dataset for LungCare AI.

Supports paired image/mask loading from:
- File-based masks  (PNG, BMP, etc.)
- Run-length encoded masks  (SIIM-ACR Pneumothorax format)

Both the image and mask pass through an Albumentations Compose pipeline
so that geometric transforms are applied identically to both tensors.

Supported datasets (after :mod:`scripts.prepare_data` preprocessing)
---------------------------------------------------------------------
- SIIM-ACR Pneumothorax  (RLE-encoded masks in CSV)
- Montgomery County  (left + right lung PNG masks, pre-merged)
- COVID-QU-Ex  (infection region PNG masks)
- MosMedData  (CT volume masks — slice-extracted by prepare_data)
- LIDC-IDRI  (nodule masks — slice-extracted by prepare_data)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from datasets.base_dataset import BaseDataset
from datasets.transforms import build_transforms

try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore[assignment]

logger = logging.getLogger("lungcare.dataset.segmentation")


# ─── RLE utilities ────────────────────────────────────────────────────────────


def rle_to_mask(
    rle_string: str | float,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Decode a SIIM-ACR style run-length encoded mask.

    Pixels are 1-indexed and enumerated in **column-major** (Fortran) order.
    A value of ``-1``, empty string, or NaN indicates no mask (all zeros).

    Args:
        rle_string: Space-separated ``start length`` pairs, or ``'-1'`` / ``NaN``.
        height: Image height in pixels (number of rows).
        width: Image width in pixels (number of columns).

    Returns:
        Binary ``uint8`` mask of shape ``(height, width)``.
    """
    if (
        rle_string is None
        or (isinstance(rle_string, float) and np.isnan(rle_string))
        or str(rle_string).strip() in ("-1", "")
    ):
        return np.zeros((height, width), dtype=np.uint8)

    parts = list(map(int, str(rle_string).split()))
    if len(parts) % 2 != 0:
        logger.warning("Odd number of RLE tokens — truncating last token.")
        parts = parts[:-1]

    pixel_count = height * width
    flat = np.zeros(pixel_count, dtype=np.uint8)
    starts, lengths = parts[0::2], parts[1::2]

    for start, length in zip(starts, lengths):
        start -= 1
        end = min(start + length, pixel_count)
        if start >= 0:
            flat[start:end] = 1

    return flat.reshape(width, height).T


def mask_to_rle(mask: np.ndarray) -> str:
    """
    Encode a binary mask as a SIIM-ACR RLE string.

    Args:
        mask: Binary ``(H, W)`` array.

    Returns:
        RLE string, or ``'-1'`` if the mask is entirely zero.
    """
    flat = mask.T.flatten()
    if flat.sum() == 0:
        return "-1"

    runs: list[int] = []
    in_run = False
    start = 0
    for i, v in enumerate(flat):
        if v and not in_run:
            start = i + 1
            in_run = True
        elif not v and in_run:
            runs.extend([start, i - start + 1])
            in_run = False
    if in_run:
        runs.extend([start, len(flat) - start + 1])
    return " ".join(map(str, runs))


# ─── Dataset ──────────────────────────────────────────────────────────────────


class SegmentationDataset(BaseDataset):
    """
    Paired image/mask segmentation dataset.

    The DataFrame must contain:
    - ``image_col``: paths to image files.
    - Either ``mask_col`` (file paths) **or** ``rle_col`` (RLE strings).

    When using ``rle_col``, you must also pass *mask_height* and *mask_width*
    so the RLE can be decoded before being resized by the transform.

    Args:
        dataframe: Source DataFrame.
        image_col: Column with image file paths.
        mask_col: Column with mask file paths.  Mutually exclusive with
            *rle_col*.
        rle_col: Column with RLE-encoded mask strings.  Mutually exclusive
            with *mask_col*.
        label_col: Optional column with integer disease class label.
        mask_height: Original mask height — required when *rle_col* is set.
        mask_width: Original mask width — required when *rle_col* is set.
        task: ``'binary'`` (default) or ``'multiclass'``.
        transform: Albumentations Compose pipeline with
            ``additional_targets={"mask": "mask"}``.
        cache: Enable sample caching.
        cache_dir: On-disk cache directory.
        image_channels: 1 for greyscale CT/CXR, 3 for RGB.
        fallback_size: ``(H, W)`` of zero image returned for corrupted files.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        image_col: str = "image_path",
        mask_col: str | None = "mask_path",
        rle_col: str | None = None,
        label_col: str | None = None,
        mask_height: int = 1024,
        mask_width: int = 1024,
        task: Literal["binary", "multiclass"] = "binary",
        transform: "A.Compose | None" = None,
        cache: bool = False,
        cache_dir: Path | str | None = None,
        image_channels: int = 1,
        fallback_size: tuple[int, int] = (256, 256),
        *,
        csv_path: Path | str | None = None,
        image_size: int | tuple[int, int] | None = None,
        split: str | None = None,
        split_col: str = "split",
        in_channels: int | None = None,
    ) -> None:
        """
        Two initialisation styles are supported (fully backward compatible):

        1. **DataFrame** (original): ``SegmentationDataset(dataframe, ...)``.
        2. **CSV manifest** (new): ``SegmentationDataset(csv_path=..., image_size=...,
           split=..., in_channels=...)``.  When *image_size* is given and no
           explicit *transform* is passed, a default segmentation pipeline
           (``Resize → Normalize → ToTensorV2`` with a paired ``mask`` target)
           is built for *split*.

        Args:
            csv_path: Path to a CSV manifest with ``image_path`` and ``mask_path``.
            image_size: Target ``int`` or ``(H, W)`` for the default transform.
            split: Split name — selects train augmentation vs deterministic
                val/test transforms (and filters rows if *split_col* exists).
            in_channels: Alias for *image_channels* (1 = greyscale, 3 = RGB).
        """
        if in_channels is not None:
            image_channels = in_channels

        # ── Resolve DataFrame from csv_path when needed ───────────────────────
        if dataframe is None:
            if csv_path is None:
                raise ValueError(
                    "SegmentationDataset requires either 'dataframe' or 'csv_path'."
                )
            df = pd.read_csv(csv_path)
            if split is not None and split_col in df.columns:
                df = df[df[split_col] == split].reset_index(drop=True)
            dataframe = df

        # ── Build a default segmentation transform when a size is provided ────
        if transform is None and image_size is not None:
            transform = build_transforms(
                split=split or "val",
                image_size=image_size,
                channels=image_channels,
                is_segmentation=True,
            )
            if fallback_size == (256, 256):
                fallback_size = (
                    (image_size, image_size)
                    if isinstance(image_size, int)
                    else (int(image_size[0]), int(image_size[1]))
                )

        super().__init__(
            transform=transform,
            cache=cache,
            cache_dir=cache_dir,
            image_channels=image_channels,
            fallback_size=fallback_size,
        )
        if mask_col is None and rle_col is None:
            raise ValueError("Provide either 'mask_col' or 'rle_col'.")
        if mask_col is not None and rle_col is not None:
            raise ValueError(
                "'mask_col' and 'rle_col' are mutually exclusive."
            )

        required_cols = [image_col]
        if mask_col:
            required_cols.append(mask_col)
        if rle_col:
            required_cols.append(rle_col)
        for col in required_cols:
            if col not in dataframe.columns:
                raise KeyError(f"Column '{col}' not found in dataframe.")

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_col = image_col
        self.mask_col = mask_col
        self.rle_col = rle_col
        self.label_col = label_col
        self.mask_height = mask_height
        self.mask_width = mask_width
        self.task = task
        self._use_rle = rle_col is not None

        logger.info(
            "SegmentationDataset | samples=%d | mask_source=%s | channels=%d",
            len(self.dataframe),
            "rle" if self._use_rle else "file",
            image_channels,
        )

    def _load_mask(self, row: pd.Series) -> np.ndarray:
        """Load or decode a mask from the row, returning binary ``(H, W)`` uint8."""
        if self._use_rle:
            return rle_to_mask(
                row[self.rle_col],  # type: ignore[index]
                self.mask_height,
                self.mask_width,
            )
        mask_path = Path(str(row[self.mask_col]))
        return self.load_mask(mask_path)

    def _get_label(self, row: pd.Series) -> torch.Tensor:
        """Extract the disease-class label (scalar long) from the row."""
        if self.label_col and self.label_col in row.index:
            return torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return torch.tensor(-1, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cached = self._get_from_cache(idx)
        if cached is not None:
            return cached

        row = self.dataframe.iloc[idx]
        image_path = Path(str(row[self.image_col]))
        label_tensor = self._get_label(row)

        mask_source = (
            str(row[self.rle_col]) if self._use_rle
            else str(row[self.mask_col])
        )

        metadata: dict[str, Any] = {
            "idx": idx,
            "image_path": str(image_path),
            "mask_source": mask_source,
            "is_corrupted": False,
        }
        if "split" in row.index:
            metadata["split"] = str(row["split"])

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = self.load_image(image_path)
            metadata["original_shape"] = (image.shape[0], image.shape[1])

            mask = self._load_mask(row)

            if mask.shape[:2] != image.shape[:2]:
                import cv2
                mask = cv2.resize(
                    mask,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            if image.ndim == 3 and image.shape[2] == 1:
                image_for_transform = image[:, :, 0]
            else:
                image_for_transform = image

            img_tensor, mask_tensor = self._apply_transform(
                image_for_transform if image_for_transform.ndim == 3 else image,
                mask,
            )

        except Exception as exc:
            logger.warning(
                "Failed to load sample idx=%d path=%s: %s", idx, image_path, exc
            )
            sample: dict[str, Any] = self._make_fallback_sample(
                idx,
                label_tensor,
                extra_metadata={
                    "image_path": str(image_path),
                    "mask_source": mask_source,
                    "is_corrupted": True,
                },
            )
            h, w = self._fallback_size
            sample["mask"] = torch.zeros(1, h, w, dtype=torch.float32)
            self._save_to_cache(idx, sample)
            return sample

        sample = {
            "image": img_tensor,
            "label": label_tensor,
            "mask": mask_tensor,
            "metadata": metadata,
        }
        self._save_to_cache(idx, sample)
        return sample

    def get_sample_info(self, idx: int) -> dict[str, Any]:
        """Return metadata for *idx* without loading image or mask."""
        row = self.dataframe.iloc[idx]
        info: dict[str, Any] = {
            "idx": idx,
            "image_path": str(row[self.image_col]),
            "mask_source": (
                str(row[self.rle_col]) if self._use_rle
                else str(row[self.mask_col])
            ),
            "use_rle": self._use_rle,
        }
        label = self._get_label(row)
        info["label_idx"] = int(label.item())
        if "split" in row.index:
            info["split"] = str(row["split"])
        return info

    def get_class_distribution(self) -> dict[str, int]:
        """Return count of masked vs non-masked samples."""
        if self._use_rle:
            has_mask = self.dataframe[self.rle_col].apply(
                lambda v: str(v).strip() not in ("-1", "", "nan")
                and not (isinstance(v, float) and np.isnan(v))
            )
        else:
            has_mask = self.dataframe[self.mask_col].notna()

        return {
            "has_mask": int(has_mask.sum()),
            "no_mask": int((~has_mask).sum()),
        }

    def get_positive_fraction(self) -> float:
        """
        Return the fraction of samples that contain a non-empty mask.

        Useful for computing Focal Loss alpha or class weighting.
        """
        dist = self.get_class_distribution()
        total = dist["has_mask"] + dist["no_mask"]
        return dist["has_mask"] / total if total > 0 else 0.0

    @classmethod
    def from_csv(
        cls,
        csv_path: Path | str,
        split: str | None = None,
        split_col: str = "split",
        **kwargs: Any,
    ) -> "SegmentationDataset":
        """
        Construct a :class:`SegmentationDataset` from a CSV file.

        Args:
            csv_path: Path to the prepared CSV.
            split: Optional split filter (``'train'``, ``'val'``, ``'test'``).
            split_col: Column to filter on.
            **kwargs: Forwarded to the constructor.
        """
        df = pd.read_csv(csv_path)
        if split is not None and split_col in df.columns:
            df = df[df[split_col] == split].reset_index(drop=True)
        return cls(df, **kwargs)
