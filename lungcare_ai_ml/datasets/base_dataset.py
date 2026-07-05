"""
Abstract base dataset for LungCare AI.

All dataset classes inherit from :class:`BaseDataset`, which provides:
- Unified ``__getitem__`` return format (image / label / mask / metadata dict).
- Image loading for PNG/JPG and DICOM files with robust error handling.
- Optional in-memory and on-disk caching.
- Sample-weight computation for :class:`torch.utils.data.WeightedRandomSampler`.
- Unit-test–friendly introspection helpers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("lungcare.dataset.base")

_DICOM_EXTENSIONS: frozenset[str] = frozenset({".dcm", ".dicom"})
_NIFTI_EXTENSIONS: frozenset[str] = frozenset({".nii", ".gz"})


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all LungCare AI datasets.

    Subclasses must implement :meth:`__len__`, :meth:`__getitem__`,
    :meth:`get_sample_info`, and :meth:`get_class_distribution`.

    Every :meth:`__getitem__` call must return a dict with the following
    top-level keys::

        {
            "image":    torch.Tensor,         # (C, H, W) float32
            "label":    torch.Tensor,         # scalar or (N,) float32
            "mask":     torch.Tensor | None,  # (1, H, W) float32
            "metadata": dict[str, Any],       # image_path, shape, etc.
        }

    Args:
        transform: Albumentations :class:`Compose` pipeline.  Must end
            with ``ToTensorV2`` so outputs are already tensors.
        cache: Whether to cache processed samples after the first load.
        cache_dir: Directory for on-disk cache.  When ``None`` and
            *cache* is ``True``, an in-memory cache is used instead.
        image_channels: Number of channels to load (1 = greyscale, 3 = RGB).
        fallback_size: ``(H, W)`` of the black replacement image returned
            when an image fails to load.
    """

    def __init__(
        self,
        transform: A.Compose | None = None,
        cache: bool = False,
        cache_dir: Path | str | None = None,
        image_channels: int = 3,
        fallback_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.transform = transform
        self.image_channels = image_channels
        self._fallback_size = fallback_size
        self._failed_indices: set[int] = set()

        self._use_disk_cache: bool = False
        self._memory_cache: dict[int, dict[str, Any]] = {}
        self._cache_dir: Path | None = None

        if cache and cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._use_disk_cache = True
            logger.info("Disk cache enabled: %s", self._cache_dir)
        elif cache:
            logger.info("In-memory cache enabled.")

        self._cache_enabled = cache

    # ─── Abstract interface ───────────────────────────────────────────────────

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return a processed sample dict for index *idx*.

        Must always return the four canonical keys:
        ``image``, ``label``, ``mask``, ``metadata``.
        """

    @abstractmethod
    def get_sample_info(self, idx: int) -> dict[str, Any]:
        """
        Return lightweight metadata for *idx* without loading the image.

        Intended for unit tests and dataset inspection tools.
        """

    @abstractmethod
    def get_class_distribution(self) -> dict[str, int]:
        """
        Return a mapping from class name to sample count.

        Used for computing class weights and reporting dataset statistics.
        """

    # ─── Image loading ────────────────────────────────────────────────────────

    def load_image(
        self,
        path: Path,
        channels: int | None = None,
    ) -> np.ndarray:
        """
        Load an image from disk.

        Supports PNG, JPG, BMP (via OpenCV) and DICOM files (via
        :mod:`utils.dicom_utils`).  Returns an ``uint8`` NumPy array
        in shape ``(H, W, C)``.

        Args:
            path: Absolute or relative path to the image file.
            channels: Override ``self.image_channels`` for this call.

        Returns:
            ``uint8`` array of shape ``(H, W, C)``.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the file cannot be decoded.
        """
        ch = channels if channels is not None else self.image_channels
        suffix = path.suffix.lower()

        if suffix in _DICOM_EXTENSIONS:
            return self._load_dicom_image(path, ch)

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"cv2.imread returned None for: {path}")

        if ch == 3:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[:, :, np.newaxis]

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)

        return image

    def _load_dicom_image(self, path: Path, channels: int) -> np.ndarray:
        """Load a DICOM file using :func:`utils.dicom_utils.dicom_to_image`."""
        from utils.dicom_utils import dicom_to_image

        image, _ = dicom_to_image(path, to_rgb=(channels == 3))
        if channels == 1 and image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
        return image

    def load_mask(self, path: Path) -> np.ndarray:
        """
        Load a binary mask from a PNG/BMP file.

        Pixels above 127 are set to 1; the rest to 0.

        Args:
            path: Path to the mask image.

        Returns:
            ``uint8`` binary array of shape ``(H, W)``.

        Raises:
            RuntimeError: If the file cannot be decoded.
        """
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not load mask: {path}")
        return (mask > 127).astype(np.uint8)

    # ─── Transform application ────────────────────────────────────────────────

    def _apply_transform(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Apply ``self.transform`` to *image* and optionally *mask*.

        Args:
            image: ``uint8`` array ``(H, W, C)`` or ``(H, W, 1)``.
            mask: Optional ``uint8`` binary array ``(H, W)``.

        Returns:
            ``(image_tensor, mask_tensor)`` where:
            - *image_tensor* is ``float32`` ``(C, H, W)``.
            - *mask_tensor* is ``float32`` ``(1, H, W)`` or ``None``.
        """
        if image.ndim == 3 and image.shape[2] == 1:
            image_3d = np.repeat(image, 3, axis=2) if self.image_channels == 3 else image
        else:
            image_3d = image

        if self.transform is not None:
            if mask is not None:
                result = self.transform(image=image_3d, mask=mask)
                img_t: torch.Tensor = result["image"]
                msk_t: torch.Tensor = result["mask"]
                if not isinstance(img_t, torch.Tensor):
                    img_t = torch.from_numpy(
                        np.transpose(img_t, (2, 0, 1)).copy()
                    ).float()
                if not isinstance(msk_t, torch.Tensor):
                    msk_t = torch.from_numpy(msk_t.copy()).float()
                mask_tensor = msk_t.unsqueeze(0).float()
                return img_t.float(), mask_tensor
            else:
                result = self.transform(image=image_3d)
                img_t = result["image"]
                if not isinstance(img_t, torch.Tensor):
                    img_t = torch.from_numpy(
                        np.transpose(img_t, (2, 0, 1)).copy()
                    ).float()
                return img_t.float(), None

        return self._numpy_to_tensor(image_3d), (
            torch.from_numpy(mask.copy()).unsqueeze(0).float()
            if mask is not None
            else None
        )

    @staticmethod
    def _numpy_to_tensor(image: np.ndarray) -> torch.Tensor:
        """Convert a ``(H, W, C)`` uint8 array to a ``(C, H, W)`` float32 tensor."""
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        return tensor / 255.0 if tensor.max() > 1.0 else tensor

    # ─── Caching ──────────────────────────────────────────────────────────────

    def _get_from_cache(self, idx: int) -> dict[str, Any] | None:
        """
        Retrieve a processed sample from the cache.

        Returns ``None`` if the sample is not in cache.
        """
        if not self._cache_enabled:
            return None

        if not self._use_disk_cache:
            return self._memory_cache.get(idx)

        cache_path = self._cache_dir / f"{idx}.pt"  # type: ignore[operator]
        if cache_path.exists():
            try:
                return torch.load(cache_path, map_location="cpu", weights_only=False)
            except Exception as exc:
                logger.warning("Cache read failed for idx=%d: %s", idx, exc)
                cache_path.unlink(missing_ok=True)
        return None

    def _save_to_cache(self, idx: int, sample: dict[str, Any]) -> None:
        """Persist a processed sample to cache."""
        if not self._cache_enabled:
            return

        if not self._use_disk_cache:
            self._memory_cache[idx] = sample
            return

        cache_path = self._cache_dir / f"{idx}.pt"  # type: ignore[operator]
        try:
            torch.save(sample, cache_path)
        except Exception as exc:
            logger.warning("Cache write failed for idx=%d: %s", idx, exc)

    # ─── Fallback handling ────────────────────────────────────────────────────

    def _make_fallback_image(self) -> np.ndarray:
        """Return a black placeholder image for corrupted files."""
        h, w = self._fallback_size
        ch = self.image_channels if self.image_channels > 1 else 1
        return np.zeros((h, w, ch), dtype=np.uint8)

    def _make_fallback_sample(
        self,
        idx: int,
        label_tensor: torch.Tensor,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build a zeroed-out fallback sample for a corrupted image.

        Args:
            idx: Dataset index of the corrupted sample.
            label_tensor: Label to include even for corrupted images.
            extra_metadata: Additional metadata key/value pairs.

        Returns:
            A valid sample dict with all four canonical keys.
        """
        self._failed_indices.add(idx)
        fallback_img = self._make_fallback_image()
        img_t, _ = self._apply_transform(fallback_img)
        metadata: dict[str, Any] = {
            "idx": idx,
            "is_corrupted": True,
            **(extra_metadata or {}),
        }
        return {
            "image": img_t,
            "label": label_tensor,
            "mask": None,
            "metadata": metadata,
        }

    # ─── Utility / introspection ──────────────────────────────────────────────

    def get_sample_weights(self) -> list[float]:
        """
        Compute per-sample weights for :class:`torch.utils.data.WeightedRandomSampler`.

        Weights are the inverse of each sample's class frequency, normalised
        so that the rarest class has weight 1.0.

        Returns:
            List of floats, one per sample, in dataset order.

        Raises:
            NotImplementedError: When the subclass does not override this method
                *and* :meth:`get_class_distribution` returns insufficient data.
        """
        dist = self.get_class_distribution()
        n_classes = len(dist)
        n_samples = sum(dist.values())
        class_weight = {
            cls: n_samples / (n_classes * count)
            for cls, count in dist.items()
            if count > 0
        }
        weights: list[float] = []
        for idx in range(len(self)):
            info = self.get_sample_info(idx)
            cls_name = info.get("class_name", "Unknown")
            weights.append(class_weight.get(cls_name, 1.0))
        return weights

    def get_failed_indices(self) -> frozenset[int]:
        """Return indices of samples that failed to load."""
        return frozenset(self._failed_indices)

    def summary(self) -> dict[str, Any]:
        """
        Return a dict summarising the dataset.

        Includes total samples, class distribution, failed indices,
        and cache status.
        """
        dist = self.get_class_distribution()
        return {
            "total_samples": len(self),
            "class_distribution": dist,
            "num_failed": len(self._failed_indices),
            "failed_indices": list(self._failed_indices),
            "cache_enabled": self._cache_enabled,
            "cache_mode": "disk" if self._use_disk_cache else "memory",
            "image_channels": self.image_channels,
            "has_transform": self.transform is not None,
        }
