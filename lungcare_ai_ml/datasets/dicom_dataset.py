"""
DICOM and CT scan dataset for LungCare AI.

Handles two modes:
- **Slice mode** (default): each sample is a single DICOM file (2D image).
  Returns a ``(C, H, W)`` tensor after windowing and resizing.
- **Series mode**: each sample is a directory containing a DICOM series
  (CT volume).  Returns a ``(D, H, W)`` tensor of windowed slices.

Supported formats
-----------------
- ``.dcm`` / ``.dicom`` — DICOM (via :mod:`utils.dicom_utils`)
- ``.nii`` / ``.nii.gz`` — NIfTI (via SimpleITK, used by MosMedData)

Supported datasets
------------------
- MosMedData (NIfTI CT volumes + binary masks)
- LIDC-IDRI  (DICOM series + nodule annotations)
- RSNA Pneumonia (DICOM CXRs — alternative to PNG mode)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from datasets.base_dataset import BaseDataset
from utils.dicom_utils import (
    WindowPreset,
    apply_windowing,
    dicom_to_image,
    get_window_preset,
    load_dicom,
    load_dicom_series,
    normalize_pixel_array,
)

try:
    import SimpleITK as sitk

    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False

try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore[assignment]

logger = logging.getLogger("lungcare.dataset.dicom")

_NIFTI_SUFFIXES: frozenset[str] = frozenset({".nii", ".gz"})


class DicomDataset(BaseDataset):
    """
    DICOM / NIfTI CT scan dataset.

    Args:
        file_paths: List of paths to ``.dcm`` files, ``.nii(.gz)`` files,
            or directories (in series mode).
        labels: Integer class label per sample.  ``None`` entries or an
            entirely ``None`` list means labels are unavailable.
        mask_paths: Optional list of mask file paths, one per sample.
            ``None`` entries are allowed (no mask for that sample).
        window_preset: CT windowing preset applied to every sample
            (e.g. ``'lung'``, ``'mediastinal'``).  Ignored for files that
            are not CT (e.g. plain radiographs).
        series_mode: When ``True``, each *file_path* is treated as a
            directory containing a DICOM series.  The sample tensor will
            be 3-D ``(D, H, W)``.
        slice_index: In series mode, return only this slice index instead
            of the full volume.  ``None`` returns the full volume.
        output_size: ``(height, width)`` to resize 2-D slices to.
            ``None`` keeps original size.  Not applied in full-volume mode.
        transform: Albumentations pipeline (applied to 2-D slices only).
        cache: Enable sample caching.
        cache_dir: On-disk cache directory.
        image_channels: Output channels for 2-D slices (1 or 3).
        fallback_size: ``(H, W)`` of the zero tensor returned on load error.
    """

    def __init__(
        self,
        file_paths: list[Path | str],
        labels: list[int | None] | None = None,
        mask_paths: list[Path | str | None] | None = None,
        window_preset: str | WindowPreset = WindowPreset.LUNG,
        series_mode: bool = False,
        slice_index: int | None = None,
        output_size: tuple[int, int] | None = (224, 224),
        transform: "A.Compose | None" = None,
        cache: bool = False,
        cache_dir: Path | str | None = None,
        image_channels: int = 3,
        fallback_size: tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__(
            transform=transform,
            cache=cache,
            cache_dir=cache_dir,
            image_channels=image_channels,
            fallback_size=fallback_size,
        )
        self.file_paths: list[Path] = [Path(p) for p in file_paths]
        self.labels: list[int | None] = (
            labels if labels is not None else [None] * len(file_paths)
        )
        self.mask_paths: list[Path | None] = (
            [Path(p) if p is not None else None for p in mask_paths]
            if mask_paths is not None
            else [None] * len(file_paths)
        )

        if len(self.labels) != len(self.file_paths):
            raise ValueError(
                f"labels length ({len(self.labels)}) != file_paths length "
                f"({len(self.file_paths)})."
            )
        if len(self.mask_paths) != len(self.file_paths):
            raise ValueError(
                f"mask_paths length ({len(self.mask_paths)}) != file_paths "
                f"length ({len(self.file_paths)})."
            )

        self.window_preset = window_preset
        self.series_mode = series_mode
        self.slice_index = slice_index
        self.output_size = output_size
        self._window_center, self._window_width = get_window_preset(window_preset)

        logger.info(
            "DicomDataset | samples=%d | series_mode=%s | window=%s",
            len(self.file_paths),
            series_mode,
            window_preset,
        )

    def _is_nifti(self, path: Path) -> bool:
        """Return True if *path* points to a NIfTI file."""
        return path.suffix.lower() in {".nii"} or str(path).endswith(".nii.gz")

    def _load_2d_slice(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Load and window a single 2-D DICOM or NIfTI slice.

        Returns:
            ``(image_uint8, dicom_meta_dict)`` where *image_uint8* is
            ``uint8`` ``(H, W, C)`` ready for the transform pipeline.
        """
        if self._is_nifti(path):
            return self._load_nifti_slice(path)

        image, metadata = dicom_to_image(
            path,
            window_preset=self.window_preset,
            output_size=self.output_size,
            to_rgb=(self.image_channels == 3),
        )
        if self.image_channels == 1 and image.ndim == 2:
            image = image[:, :, np.newaxis]

        meta_dict: dict[str, Any] = {
            "patient_id": metadata.patient_id,
            "modality": metadata.modality,
            "study_date": metadata.study_date,
            "original_shape": (metadata.rows, metadata.columns),
            "window_preset": str(self.window_preset),
            "is_series": False,
        }
        return image, meta_dict

    def _load_nifti_slice(
        self, path: Path
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Load a NIfTI volume and return the central axial slice as uint8."""
        if not _HAS_SITK:
            raise RuntimeError(
                "SimpleITK is required to load NIfTI files. "
                "Install it with: pip install SimpleITK"
            )
        img = sitk.ReadImage(str(path))
        volume: np.ndarray = sitk.GetArrayFromImage(img).astype(np.float32)
        windowed = apply_windowing(volume, self._window_center, self._window_width)

        slice_idx = self.slice_index if self.slice_index is not None else volume.shape[0] // 2
        slice_idx = max(0, min(slice_idx, volume.shape[0] - 1))
        slc = windowed[slice_idx]

        normalised = normalize_pixel_array(slc).astype(np.uint8)
        if self.output_size is not None:
            import cv2
            normalised = cv2.resize(
                normalised, (self.output_size[1], self.output_size[0]),
                interpolation=cv2.INTER_AREA,
            )

        if self.image_channels == 3:
            import cv2
            normalised = cv2.cvtColor(normalised, cv2.COLOR_GRAY2RGB)
        else:
            normalised = normalised[:, :, np.newaxis]

        meta_dict: dict[str, Any] = {
            "patient_id": "UNKNOWN",
            "modality": "CT",
            "study_date": "",
            "original_shape": tuple(volume.shape),
            "window_preset": str(self.window_preset),
            "is_series": False,
            "slice_index": slice_idx,
            "volume_depth": volume.shape[0],
        }
        return normalised, meta_dict

    def _load_series(self, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Load a full DICOM series or NIfTI volume as a 3-D float32 array.

        Returns:
            ``(volume, meta_dict)`` where *volume* has shape ``(D, H, W)``.
        """
        if self._is_nifti(path):
            if not _HAS_SITK:
                raise RuntimeError("SimpleITK is required for NIfTI volumes.")
            img = sitk.ReadImage(str(path))
            volume: np.ndarray = sitk.GetArrayFromImage(img).astype(np.float32)
        elif path.is_dir():
            volume = load_dicom_series(path, window_preset=self.window_preset)
        else:
            pixel_arr, _ = load_dicom(path)
            volume = pixel_arr[np.newaxis, ...]

        windowed = apply_windowing(volume, self._window_center, self._window_width)
        volume_norm = normalize_pixel_array(windowed, out_min=0.0, out_max=1.0)

        if self.slice_index is not None:
            si = max(0, min(self.slice_index, volume_norm.shape[0] - 1))
            volume_norm = volume_norm[si : si + 1]

        meta_dict: dict[str, Any] = {
            "patient_id": "UNKNOWN",
            "modality": "CT",
            "study_date": "",
            "original_shape": tuple(volume.shape),
            "window_preset": str(self.window_preset),
            "is_series": True,
            "depth": volume_norm.shape[0],
        }
        return volume_norm, meta_dict

    def _load_mask_volume(self, mask_path: Path) -> np.ndarray | None:
        """Load an optional mask (2-D file or NIfTI volume)."""
        try:
            if self._is_nifti(mask_path):
                if not _HAS_SITK:
                    return None
                img = sitk.ReadImage(str(mask_path))
                vol: np.ndarray = sitk.GetArrayFromImage(img)
                return (vol > 0).astype(np.uint8)
            return self.load_mask(mask_path)
        except Exception as exc:
            logger.warning("Failed to load mask %s: %s", mask_path, exc)
            return None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cached = self._get_from_cache(idx)
        if cached is not None:
            return cached

        path = self.file_paths[idx]
        raw_label = self.labels[idx]
        label_tensor = torch.tensor(
            raw_label if raw_label is not None else -1, dtype=torch.long
        )
        mask_path = self.mask_paths[idx]

        metadata: dict[str, Any] = {
            "idx": idx,
            "dicom_path": str(path),
            "window_preset": str(self.window_preset),
            "is_corrupted": False,
        }

        try:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if self.series_mode:
                volume, meta_extra = self._load_series(path)
                metadata.update(meta_extra)
                volume_tensor = torch.from_numpy(volume.copy()).float()
                mask_tensor: torch.Tensor | None = None
                if mask_path is not None:
                    mask_arr = self._load_mask_volume(mask_path)
                    if mask_arr is not None:
                        mask_tensor = torch.from_numpy(
                            mask_arr.copy()
                        ).float()
                sample = {
                    "image": volume_tensor,
                    "label": label_tensor,
                    "mask": mask_tensor,
                    "metadata": metadata,
                }
            else:
                image, meta_extra = self._load_2d_slice(path)
                metadata.update(meta_extra)
                metadata["original_shape"] = (image.shape[0], image.shape[1])

                mask_arr_2d: np.ndarray | None = None
                if mask_path is not None:
                    loaded_mask = self._load_mask_volume(mask_path)
                    if loaded_mask is not None and loaded_mask.ndim == 2:
                        mask_arr_2d = loaded_mask

                img_tensor, mask_tensor = self._apply_transform(image, mask_arr_2d)
                sample = {
                    "image": img_tensor,
                    "label": label_tensor,
                    "mask": mask_tensor,
                    "metadata": metadata,
                }

        except Exception as exc:
            logger.warning(
                "Failed to load DICOM idx=%d path=%s: %s", idx, path, exc
            )
            sample = self._make_fallback_sample(
                idx,
                label_tensor,
                extra_metadata={"dicom_path": str(path), "is_corrupted": True},
            )

        self._save_to_cache(idx, sample)
        return sample

    def get_sample_info(self, idx: int) -> dict[str, Any]:
        """Return lightweight metadata for *idx* without loading data."""
        path = self.file_paths[idx]
        return {
            "idx": idx,
            "dicom_path": str(path),
            "label": self.labels[idx],
            "has_mask": self.mask_paths[idx] is not None,
            "is_series": self.series_mode or path.is_dir(),
            "window_preset": str(self.window_preset),
        }

    def get_class_distribution(self) -> dict[str, int]:
        """Return count per integer class label (``-1`` = unlabelled)."""
        from collections import Counter

        counter: Counter = Counter(
            lbl if lbl is not None else -1 for lbl in self.labels
        )
        return {str(k): v for k, v in counter.items()}

    @classmethod
    def from_directory(
        cls,
        directory: Path | str,
        pattern: str = "**/*.dcm",
        labels: list[int | None] | None = None,
        **kwargs: Any,
    ) -> "DicomDataset":
        """
        Build a :class:`DicomDataset` by globbing DICOM files from a directory.

        Args:
            directory: Root directory to search.
            pattern: Glob pattern relative to *directory*.
            labels: Optional labels; must match the number of discovered files.
            **kwargs: Forwarded to the constructor.

        Returns:
            A :class:`DicomDataset` instance.
        """
        root = Path(directory)
        paths = sorted(root.glob(pattern))
        if not paths:
            raise RuntimeError(
                f"No files matching '{pattern}' found under: {root}"
            )
        if labels is not None and len(labels) != len(paths):
            raise ValueError(
                f"labels length ({len(labels)}) != discovered files "
                f"({len(paths)})."
            )
        logger.info("DicomDataset.from_directory: found %d files.", len(paths))
        return cls(paths, labels=labels, **kwargs)
