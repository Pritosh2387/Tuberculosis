"""
Multi-disease chest X-ray classification dataset for LungCare AI.

Accepts a :class:`pandas.DataFrame` with an ``image_path`` column and one
or more label columns.  Supports binary, multiclass, and multilabel tasks
without modifying the data loading logic — only the label tensor shape and
dtype changes.

Supported tasks
---------------
- ``'binary'``     — label is a scalar ``long`` tensor (0 or 1).
- ``'multiclass'`` — label is a scalar ``long`` tensor (class index 0..N-1).
- ``'multilabel'`` — label is a ``float32`` vector of length N (one-hot or soft).

Dataset sources (after :mod:`scripts.prepare_data` preprocessing)
------------------------------------------------------------------
- NIH ChestX-ray14
- Montgomery County TB dataset
- Shenzhen TB dataset
- RSNA Pneumonia Detection Challenge
- COVID-QU-Ex Dataset
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from datasets.base_dataset import BaseDataset
from datasets.transforms import build_transforms, get_identity_transform

try:
    import albumentations as A
except ImportError:
    A = None  # type: ignore[assignment]

logger = logging.getLogger("lungcare.dataset.classification")

_DEFAULT_CLASSES = [
    "Healthy",
    "Tuberculosis",
    "Pneumonia",
    "COVID-19",
    "Lung Cancer",
    "Pulmonary Fibrosis",
]


def _default_classes(num_classes: int) -> list[str]:
    """
    Return an ordered class list of length *num_classes*.

    Uses the canonical LungCare disease order for the first six classes so
    string labels such as ``"Healthy"`` / ``"Tuberculosis"`` map to stable
    indices, then falls back to numeric names for any extra classes.
    """
    if num_classes <= len(_DEFAULT_CLASSES):
        return _DEFAULT_CLASSES[:num_classes]
    extra = [str(i) for i in range(len(_DEFAULT_CLASSES), num_classes)]
    return _DEFAULT_CLASSES + extra


class ClassificationDataset(BaseDataset):
    """
    Chest X-ray multi-disease classification dataset.

    Args:
        dataframe: DataFrame containing at minimum an ``image_path`` column
            and the columns specified by *label_col*.
        image_col: Column name containing image file paths.
        label_col: For ``'binary'`` / ``'multiclass'``: name of the column
            holding integer or string class labels.  For ``'multilabel'``:
            a list of binary column names, one per class.
        classes: Ordered list of class names.  If ``None``, inferred from
            unique values in the label column.
        task: One of ``'binary'``, ``'multiclass'``, ``'multilabel'``.
        transform: Albumentations :class:`Compose` pipeline.
        cache: Whether to cache processed samples.
        cache_dir: On-disk cache directory.
        image_channels: 1 for greyscale, 3 for RGB.
        fallback_size: ``(H, W)`` of the zero image returned for corrupted
            files.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        image_col: str = "image_path",
        label_col: str | list[str] = "label",
        classes: list[str] | None = None,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass",
        transform: "A.Compose | None" = None,
        cache: bool = False,
        cache_dir: Path | str | None = None,
        image_channels: int = 3,
        fallback_size: tuple[int, int] = (224, 224),
        *,
        csv_path: Path | str | None = None,
        image_size: int | tuple[int, int] | None = None,
        split: str | None = None,
        split_col: str = "split",
        num_classes: int | None = None,
        class_names: list[str] | None = None,
        in_channels: int | None = None,
    ) -> None:
        """
        Two initialisation styles are supported (fully backward compatible):

        1. **DataFrame** (original): ``ClassificationDataset(dataframe, ...)``.
        2. **CSV manifest** (new): ``ClassificationDataset(csv_path=..., image_size=...,
           split=..., num_classes=..., task=..., class_names=...)``.  When
           *image_size* is given and no explicit *transform* is passed, a default
           ``Resize → Normalize → ToTensorV2`` pipeline is built for *split*.

        Args:
            csv_path: Path to a CSV manifest.  Loaded into a DataFrame; rows are
                filtered to *split* only when a *split_col* column is present.
            image_size: Target ``int`` or ``(H, W)`` used to build a default
                transform when *transform* is ``None``.
            split: Split name — selects train augmentation vs deterministic
                val/test transforms (and filters rows if *split_col* exists).
            num_classes: Number of classes; when *class_names* is omitted the
                canonical LungCare class order is used to build the label map.
            class_names: Explicit ordered class names (alias for *classes*).
            in_channels: Alias for *image_channels*.
        """
        if in_channels is not None:
            image_channels = in_channels

        # ── Resolve DataFrame from csv_path when needed ───────────────────────
        if dataframe is None:
            if csv_path is None:
                raise ValueError(
                    "ClassificationDataset requires either 'dataframe' or 'csv_path'."
                )
            df = pd.read_csv(csv_path)
            if split is not None and split_col in df.columns:
                df = df[df[split_col] == split].reset_index(drop=True)
            dataframe = df

        # ── Resolve class list from class_names / num_classes ─────────────────
        if classes is None:
            if class_names is not None:
                classes = list(class_names)
            elif num_classes is not None:
                classes = _default_classes(num_classes)

        # ── Build a default transform when an image size is provided ──────────
        if transform is None and image_size is not None:
            transform = build_transforms(
                split=split or "val",
                image_size=image_size,
                channels=image_channels,
                is_segmentation=False,
            )
            if fallback_size == (224, 224):
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
        if image_col not in dataframe.columns:
            raise KeyError(f"Column '{image_col}' not found in dataframe.")

        self.dataframe = dataframe.reset_index(drop=True)
        self.image_col = image_col
        self.task = task

        if isinstance(label_col, str):
            self.label_cols: list[str] = [label_col]
        else:
            self.label_cols = list(label_col)

        for col in self.label_cols:
            if col not in dataframe.columns:
                raise KeyError(f"Label column '{col}' not found in dataframe.")

        self.classes = classes or self._infer_classes()
        self._cls_to_idx: dict[str, int] = {
            c: i for i, c in enumerate(self.classes)
        }
        self._idx_to_cls: dict[int, str] = {
            i: c for c, i in self._cls_to_idx.items()
        }

        logger.info(
            "ClassificationDataset | task=%s | samples=%d | classes=%s",
            task,
            len(self.dataframe),
            self.classes,
        )

    def _infer_classes(self) -> list[str]:
        """Infer ordered class list from the label column(s)."""
        if self.task == "multilabel":
            return self.label_cols
        col = self.label_cols[0]
        unique = sorted(self.dataframe[col].dropna().unique())
        if all(isinstance(v, (int, np.integer)) for v in unique):
            return [str(v) for v in unique]
        return [str(v) for v in unique]

    def _get_label_tensor(self, idx: int) -> tuple[torch.Tensor, str | list[str]]:
        """
        Encode the label at *idx* as a tensor.

        Returns:
            ``(label_tensor, class_name)`` where *class_name* is a string
            for single-label tasks or a list of active class names for
            multilabel tasks.
        """
        row = self.dataframe.iloc[idx]

        if self.task == "multilabel":
            values = [float(row[col]) for col in self.label_cols]
            label_tensor = torch.tensor(values, dtype=torch.float32)
            active = [
                self.label_cols[i] for i, v in enumerate(values) if v > 0.5
            ]
            return label_tensor, active

        raw = row[self.label_cols[0]]
        if isinstance(raw, (int, np.integer)):
            idx_val = int(raw)
            cls_name = self._idx_to_cls.get(idx_val, str(idx_val))
        elif isinstance(raw, str):
            idx_val = self._cls_to_idx.get(raw, 0)
            cls_name = raw
        else:
            idx_val = int(raw)
            cls_name = self._idx_to_cls.get(idx_val, str(idx_val))

        return torch.tensor(idx_val, dtype=torch.long), cls_name

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cached = self._get_from_cache(idx)
        if cached is not None:
            return cached

        label_tensor, class_name = self._get_label_tensor(idx)
        row = self.dataframe.iloc[idx]
        image_path = Path(str(row[self.image_col]))

        metadata: dict[str, Any] = {
            "idx": idx,
            "image_path": str(image_path),
            "class_name": class_name,
            "is_corrupted": False,
        }

        if "split" in row.index:
            metadata["split"] = str(row["split"])
        if "patient_id" in row.index:
            metadata["patient_id"] = str(row["patient_id"])

        try:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = self.load_image(image_path)
            metadata["original_shape"] = (image.shape[0], image.shape[1])
            img_tensor, _ = self._apply_transform(image, mask=None)

        except Exception as exc:
            logger.warning(
                "Failed to load sample idx=%d path=%s: %s", idx, image_path, exc
            )
            sample = self._make_fallback_sample(
                idx,
                label_tensor,
                extra_metadata={
                    "image_path": str(image_path),
                    "class_name": class_name,
                    "is_corrupted": True,
                },
            )
            # Classification samples carry no mask; a ``None`` value would break
            # ``torch.utils.data.default_collate`` when batched by a DataLoader.
            sample.pop("mask", None)
            self._save_to_cache(idx, sample)
            return sample

        sample: dict[str, Any] = {
            "image": img_tensor,
            "label": label_tensor,
            "metadata": metadata,
        }
        self._save_to_cache(idx, sample)
        return sample

    def get_sample_info(self, idx: int) -> dict[str, Any]:
        """
        Return metadata for sample *idx* without loading the image.

        Args:
            idx: Dataset index.

        Returns:
            Dict with ``image_path``, ``class_name``, ``label_idx``, ``split``.
        """
        row = self.dataframe.iloc[idx]
        label_tensor, class_name = self._get_label_tensor(idx)
        info: dict[str, Any] = {
            "idx": idx,
            "image_path": str(row[self.image_col]),
            "class_name": class_name,
        }
        if self.task != "multilabel":
            info["label_idx"] = int(label_tensor.item())
        if "split" in row.index:
            info["split"] = str(row["split"])
        return info

    def get_class_distribution(self) -> dict[str, int]:
        """
        Return the sample count per class.

        For multilabel tasks, counts the total number of positive
        labels per class (a sample contributes to multiple classes).
        """
        if self.task == "multilabel":
            return {
                col: int(self.dataframe[col].sum())
                for col in self.label_cols
            }

        col = self.label_cols[0]
        counter: Counter = Counter()
        for raw in self.dataframe[col]:
            if isinstance(raw, str):
                counter[raw] += 1
            else:
                counter[self._idx_to_cls.get(int(raw), str(int(raw)))] += 1
        return dict(counter)

    def get_sample_weights(self) -> list[float]:
        """
        Compute inverse-frequency weights for :class:`WeightedRandomSampler`.

        For multilabel tasks, each sample's weight is the inverse of the
        frequency of its rarest active class.
        """
        dist = self.get_class_distribution()
        n_total = len(self.dataframe)
        n_classes = len(dist)

        inv_freq: dict[str, float] = {
            cls: n_total / (n_classes * max(count, 1))
            for cls, count in dist.items()
        }

        weights: list[float] = []
        for idx in range(len(self)):
            _, class_name = self._get_label_tensor(idx)
            if isinstance(class_name, list):
                sample_weight = max(
                    (inv_freq.get(cn, 1.0) for cn in class_name),
                    default=1.0,
                )
            else:
                sample_weight = inv_freq.get(class_name, 1.0)
            weights.append(sample_weight)
        return weights

    @classmethod
    def from_csv(
        cls,
        csv_path: Path | str,
        image_col: str = "image_path",
        label_col: str | list[str] = "label",
        split: str | None = None,
        split_col: str = "split",
        **kwargs: Any,
    ) -> "ClassificationDataset":
        """
        Construct a :class:`ClassificationDataset` from a CSV file.

        Args:
            csv_path: Path to the CSV file.
            image_col: Column with image paths.
            label_col: Column(s) with labels.
            split: If given, filter rows where *split_col* equals *split*.
            split_col: Column to filter by split.
            **kwargs: Forwarded to the constructor.

        Returns:
            A :class:`ClassificationDataset` instance.
        """
        df = pd.read_csv(csv_path)
        if split is not None and split_col in df.columns:
            df = df[df[split_col] == split].reset_index(drop=True)
        return cls(df, image_col=image_col, label_col=label_col, **kwargs)
