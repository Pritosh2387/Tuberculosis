"""
DICOM and medical image loading utilities for LungCare AI.

Handles DICOM file reading (via ``pydicom``), Hounsfield Unit windowing,
pixel normalisation, and CT series loading (via ``SimpleITK``).  The
``dicom_to_image`` function provides a single-call pipeline from raw
DICOM to a NumPy array ready for model input.

Platform independence:  All paths are handled via :mod:`pathlib`.
No OS-specific APIs are used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydicom
import pydicom.errors

logger = logging.getLogger("lungcare.dicom")

try:
    import SimpleITK as sitk

    _HAS_SITK = True
except ImportError:
    _HAS_SITK = False
    logger.warning(
        "SimpleITK not available — CT series loading will be disabled."
    )


# ─── Window presets ───────────────────────────────────────────────────────────


class WindowPreset(str, Enum):
    """
    Standard CT windowing presets.

    Each member maps to a ``(window_center, window_width)`` pair
    measured in Hounsfield Units (HU).
    """

    LUNG = "lung"
    MEDIASTINAL = "mediastinal"
    BONE = "bone"
    BRAIN = "brain"
    ABDOMEN = "abdomen"
    SOFT_TISSUE = "soft_tissue"


_WINDOW_PRESETS: dict[WindowPreset, tuple[float, float]] = {
    WindowPreset.LUNG: (-600.0, 1500.0),
    WindowPreset.MEDIASTINAL: (40.0, 400.0),
    WindowPreset.BONE: (400.0, 1800.0),
    WindowPreset.BRAIN: (40.0, 80.0),
    WindowPreset.ABDOMEN: (60.0, 400.0),
    WindowPreset.SOFT_TISSUE: (50.0, 350.0),
}


# ─── Metadata dataclass ───────────────────────────────────────────────────────


@dataclass
class DicomMetadata:
    """
    Structured metadata extracted from a DICOM dataset.

    All fields default gracefully when the corresponding DICOM tag is absent
    (e.g. radiograph files may omit ``PixelSpacing`` or ``WindowCenter``).
    """

    patient_id: str
    study_date: str
    modality: str
    rows: int
    columns: int
    pixel_spacing: tuple[float, float] | None
    window_center: float | None
    window_width: float | None
    rescale_intercept: float
    rescale_slope: float
    bits_stored: int
    photometric_interpretation: str

    @property
    def aspect_ratio(self) -> float | None:
        """Pixel aspect ratio (row spacing / column spacing)."""
        if self.pixel_spacing is None:
            return None
        row_sp, col_sp = self.pixel_spacing
        return row_sp / col_sp if col_sp != 0 else None


# ─── Core helpers ─────────────────────────────────────────────────────────────


def _safe_float(dataset: pydicom.Dataset, tag: str, default: float) -> float:
    """Safely read a numeric DICOM tag, returning *default* if absent."""
    val = getattr(dataset, tag, None)
    if val is None:
        return default
    try:
        if hasattr(val, "__len__") and not isinstance(val, str):
            return float(val[0])
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_str(dataset: pydicom.Dataset, tag: str, default: str = "") -> str:
    """Safely read a string DICOM tag, returning *default* if absent."""
    val = getattr(dataset, tag, None)
    return str(val).strip() if val is not None else default


# ─── Public API ───────────────────────────────────────────────────────────────


def load_dicom(path: Path | str) -> tuple[np.ndarray, DicomMetadata]:
    """
    Load a DICOM file and apply RescaleSlope / RescaleIntercept.

    Args:
        path: Path to a ``.dcm`` file.

    Returns:
        A tuple ``(pixel_array, metadata)`` where *pixel_array* contains
        Hounsfield Units (for CT) or raw pixel values (for radiographs).
        Shape is ``(H, W)`` for single-frame files.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydicom.errors.InvalidDicomError: If the file is not valid DICOM.
    """
    dicom_path = Path(path)
    if not dicom_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {dicom_path}")

    ds = pydicom.dcmread(str(dicom_path))
    pixel_array = ds.pixel_array.astype(np.float32)

    slope = _safe_float(ds, "RescaleSlope", default=1.0)
    intercept = _safe_float(ds, "RescaleIntercept", default=0.0)
    pixel_array = pixel_array * slope + intercept

    pixel_spacing_tag = getattr(ds, "PixelSpacing", None)
    pixel_spacing: tuple[float, float] | None = None
    if pixel_spacing_tag is not None and len(pixel_spacing_tag) >= 2:
        pixel_spacing = (float(pixel_spacing_tag[0]), float(pixel_spacing_tag[1]))

    metadata = DicomMetadata(
        patient_id=_safe_str(ds, "PatientID", "UNKNOWN"),
        study_date=_safe_str(ds, "StudyDate", ""),
        modality=_safe_str(ds, "Modality", ""),
        rows=int(getattr(ds, "Rows", pixel_array.shape[0])),
        columns=int(getattr(ds, "Columns", pixel_array.shape[1])),
        pixel_spacing=pixel_spacing,
        window_center=_safe_float(ds, "WindowCenter", default=float("nan")) or None,
        window_width=_safe_float(ds, "WindowWidth", default=float("nan")) or None,
        rescale_intercept=intercept,
        rescale_slope=slope,
        bits_stored=int(getattr(ds, "BitsStored", 16)),
        photometric_interpretation=_safe_str(
            ds, "PhotometricInterpretation", "MONOCHROME2"
        ),
    )

    if "MONOCHROME1" in metadata.photometric_interpretation:
        max_val = pixel_array.max()
        pixel_array = max_val - pixel_array
        logger.debug("MONOCHROME1 image inverted to MONOCHROME2 convention.")

    return pixel_array, metadata


def apply_windowing(
    pixel_array: np.ndarray,
    window_center: float,
    window_width: float,
) -> np.ndarray:
    """
    Clamp a pixel array to a CT window (Hounsfield Unit range).

    Args:
        pixel_array: Float array of HU values, shape ``(H, W)``.
        window_center: Centre of the HU window.
        window_width: Width of the HU window.

    Returns:
        Array clamped to ``[center - width/2, center + width/2]``,
        same dtype as input.
    """
    low = window_center - window_width / 2.0
    high = window_center + window_width / 2.0
    return np.clip(pixel_array, low, high)


def get_window_preset(preset: str | WindowPreset) -> tuple[float, float]:
    """
    Look up a windowing preset by name.

    Args:
        preset: Either a :class:`WindowPreset` member or its string value
            (e.g. ``'lung'``, ``'bone'``).

    Returns:
        ``(window_center, window_width)`` tuple.

    Raises:
        ValueError: If the preset name is not recognised.
    """
    try:
        key = WindowPreset(preset) if isinstance(preset, str) else preset
    except ValueError:
        valid = [p.value for p in WindowPreset]
        raise ValueError(
            f"Unknown window preset '{preset}'. Valid options: {valid}."
        )
    return _WINDOW_PRESETS[key]


def normalize_pixel_array(
    pixel_array: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 255.0,
) -> np.ndarray:
    """
    Min-max normalise a pixel array to a target range.

    Args:
        pixel_array: Input float array.
        out_min: Minimum value in the output range.
        out_max: Maximum value in the output range.

    Returns:
        Normalised ``float32`` array in ``[out_min, out_max]``.
    """
    arr = pixel_array.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-8:
        return np.full_like(arr, out_min)
    normalised = (arr - vmin) / (vmax - vmin)
    return (normalised * (out_max - out_min) + out_min).astype(np.float32)


def dicom_to_image(
    path: Path | str,
    window_preset: str | WindowPreset = WindowPreset.LUNG,
    output_size: tuple[int, int] | None = None,
    to_rgb: bool = True,
    use_dicom_window: bool = False,
) -> tuple[np.ndarray, DicomMetadata]:
    """
    Full pipeline: DICOM → windowing → normalise → resize → ``uint8`` image.

    Args:
        path: Path to a ``.dcm`` file.
        window_preset: CT window preset name or :class:`WindowPreset` member.
            Ignored when *use_dicom_window* is ``True`` and the file contains
            valid ``WindowCenter`` / ``WindowWidth`` tags.
        output_size: ``(width, height)`` to resize to.  No resize if ``None``.
        to_rgb: If ``True``, convert the greyscale image to 3-channel RGB.
        use_dicom_window: If ``True``, prefer the window values embedded in
            the DICOM header over the *window_preset*.

    Returns:
        ``(image_array, metadata)`` where *image_array* is ``uint8``
        with shape ``(H, W, 3)`` when *to_rgb* is ``True``, else ``(H, W)``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    pixel_array, metadata = load_dicom(path)

    if use_dicom_window and metadata.window_center and metadata.window_width:
        center = metadata.window_center
        width = metadata.window_width
        logger.debug(
            "Using DICOM header window: center=%.1f, width=%.1f", center, width
        )
    else:
        center, width = get_window_preset(window_preset)

    windowed = apply_windowing(pixel_array, center, width)
    normalised = normalize_pixel_array(windowed, out_min=0.0, out_max=255.0)
    image = normalised.astype(np.uint8)

    if output_size is not None:
        image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)

    if to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image, metadata


def load_dicom_series(
    directory: Path | str,
    window_preset: str | WindowPreset = WindowPreset.LUNG,
    sort_by_instance: bool = True,
) -> np.ndarray:
    """
    Load a directory of DICOM slices as a 3D NumPy volume.

    Uses ``SimpleITK`` for robust series reading when available, falling
    back to ``pydicom``-based slice-by-slice loading otherwise.

    Args:
        directory: Path to a directory containing ``.dcm`` slice files.
        window_preset: CT window preset applied to every slice.
        sort_by_instance: Sort slices by ``InstanceNumber`` tag when using
            the pydicom fallback path.

    Returns:
        Float32 array of shape ``(D, H, W)`` containing HU values.

    Raises:
        NotADirectoryError: If *directory* is not a valid directory.
        RuntimeError: If no DICOM files are found in *directory*.
    """
    series_dir = Path(directory)
    if not series_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {series_dir}")

    if _HAS_SITK:
        return _load_series_sitk(series_dir, window_preset)
    return _load_series_pydicom(series_dir, window_preset, sort_by_instance)


def _load_series_sitk(
    series_dir: Path,
    window_preset: str | WindowPreset,
) -> np.ndarray:
    """Load a CT series with SimpleITK (preferred path)."""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not dicom_names:
        raise RuntimeError(f"No DICOM series found in: {series_dir}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    volume: np.ndarray = sitk.GetArrayFromImage(image).astype(np.float32)

    center, width = get_window_preset(window_preset)
    volume = apply_windowing(volume, center, width)
    logger.info(
        "Loaded CT series via SimpleITK: %s slices, shape=%s",
        volume.shape[0],
        volume.shape,
    )
    return volume


def _load_series_pydicom(
    series_dir: Path,
    window_preset: str | WindowPreset,
    sort_by_instance: bool,
) -> np.ndarray:
    """Load a CT series slice-by-slice with pydicom (fallback path)."""
    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        dcm_files = sorted(series_dir.glob("*.DCM"))
    if not dcm_files:
        raise RuntimeError(f"No .dcm files found in: {series_dir}")

    slices: list[Any] = []
    datasets: list[Any] = []
    for f in dcm_files:
        try:
            ds = pydicom.dcmread(str(f))
            datasets.append(ds)
        except pydicom.errors.InvalidDicomError:
            logger.warning("Skipping invalid DICOM file: %s", f.name)

    if sort_by_instance:
        datasets.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))

    center, width = get_window_preset(window_preset)
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = _safe_float(ds, "RescaleSlope", 1.0)
        intercept = _safe_float(ds, "RescaleIntercept", 0.0)
        arr = arr * slope + intercept
        arr = apply_windowing(arr, center, width)
        slices.append(arr)

    volume = np.stack(slices, axis=0)
    logger.info(
        "Loaded CT series via pydicom: %d slices, shape=%s",
        len(slices),
        volume.shape,
    )
    return volume
