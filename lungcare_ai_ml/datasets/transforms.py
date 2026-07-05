"""
Albumentations pipeline builder for LungCare AI.

Reads the ``{name, params}`` list format from ``augmentation_config.yaml``
and dynamically instantiates each transform via ``getattr(albumentations, name)``.
``ToTensorV2`` is handled as a special case since it lives in a sub-module.

Segmentation pipelines are built with ``additional_targets={"mask": "mask"}``
so that identical geometric transforms are applied to both image and mask.
"""

from __future__ import annotations

import logging
from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.config import AugmentationConfig, AugmentationSplitConfig, TransformConfig

logger = logging.getLogger("lungcare.transforms")

_PYTORCH_TRANSFORMS: frozenset[str] = frozenset({"ToTensorV2"})


def _instantiate_transform(tc: TransformConfig) -> A.BasicTransform:
    """
    Instantiate a single Albumentations transform from a :class:`TransformConfig`.

    Args:
        tc: A ``{name, params}`` config object.

    Returns:
        An instantiated Albumentations transform.

    Raises:
        ValueError: If *tc.name* is not found in the Albumentations namespace.
        TypeError: If *tc.params* contains arguments invalid for the transform.
    """
    if tc.name in _PYTORCH_TRANSFORMS:
        return ToTensorV2(**tc.params)

    transform_cls = getattr(A, tc.name, None)
    if transform_cls is None:
        raise ValueError(
            f"Unknown Albumentations transform: '{tc.name}'. "
            "Check augmentation_config.yaml and your albumentations version."
        )

    try:
        return transform_cls(**tc.params)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate '{tc.name}' with params {tc.params}: {exc}"
        ) from exc


def build_transform(
    transform_configs: list[TransformConfig],
    is_segmentation: bool = False,
    additional_targets: dict[str, str] | None = None,
) -> A.Compose:
    """
    Build an :class:`albumentations.Compose` pipeline from a config list.

    Args:
        transform_configs: Ordered list of :class:`TransformConfig` objects.
        is_segmentation: When ``True``, adds ``additional_targets={"mask": "mask"}``
            so geometric transforms are applied identically to image and mask.
        additional_targets: Extra target mappings merged into the Compose kwargs.

    Returns:
        A ready-to-call :class:`albumentations.Compose` object.

    Raises:
        ValueError: If any transform name is not found.
        TypeError: If any transform receives invalid keyword arguments.
    """
    transforms: list[A.BasicTransform] = []
    for tc in transform_configs:
        inst = _instantiate_transform(tc)
        transforms.append(inst)
        logger.debug("Registered transform: %s(%s)", tc.name, tc.params)

    compose_kwargs: dict[str, Any] = {}
    if is_segmentation:
        targets = {"mask": "mask"}
        if additional_targets:
            targets.update(additional_targets)
        compose_kwargs["additional_targets"] = targets

    return A.Compose(transforms, **compose_kwargs)


def _pick_split(
    split_cfg: AugmentationSplitConfig,
    split: str,
) -> list[TransformConfig]:
    """Return the transform list for *split* from *split_cfg*."""
    mapping: dict[str, list[TransformConfig]] = {
        "train": split_cfg.train,
        "val": split_cfg.val,
        "test": split_cfg.test,
    }
    if split not in mapping:
        raise ValueError(
            f"Invalid split '{split}'. Must be one of: 'train', 'val', 'test'."
        )
    transform_list = mapping[split]
    if not transform_list:
        logger.warning(
            "Empty transform list for split '%s'. "
            "Images will pass through without augmentation.",
            split,
        )
    return transform_list


def get_classification_transforms(
    aug_config: AugmentationConfig,
    split: str,
) -> A.Compose:
    """
    Build the classification transform pipeline for *split*.

    Args:
        aug_config: Full :class:`AugmentationConfig` loaded from YAML.
        split: ``'train'``, ``'val'``, or ``'test'``.

    Returns:
        Image-only :class:`albumentations.Compose` pipeline.
    """
    transform_list = _pick_split(aug_config.classification, split)
    pipeline = build_transform(transform_list, is_segmentation=False)
    logger.info(
        "Classification %s pipeline: %d transforms.", split, len(transform_list)
    )
    return pipeline


def get_segmentation_transforms(
    aug_config: AugmentationConfig,
    split: str,
) -> A.Compose:
    """
    Build the segmentation transform pipeline for *split*.

    The returned :class:`albumentations.Compose` applies geometric transforms
    identically to both ``image`` and ``mask`` keys.

    Args:
        aug_config: Full :class:`AugmentationConfig` loaded from YAML.
        split: ``'train'``, ``'val'``, or ``'test'``.

    Returns:
        Image+mask :class:`albumentations.Compose` pipeline.
    """
    transform_list = _pick_split(aug_config.segmentation, split)
    pipeline = build_transform(transform_list, is_segmentation=True)
    logger.info(
        "Segmentation %s pipeline: %d transforms.", split, len(transform_list)
    )
    return pipeline


def get_identity_transform(
    image_size: int = 224,
    channels: int = 3,
    is_segmentation: bool = False,
) -> A.Compose:
    """
    Build a minimal fallback pipeline when no config is available.

    Performs only ``Resize → Normalize → ToTensorV2``.

    Args:
        image_size: Target square size (height = width).
        channels: Number of image channels (1 or 3).
        is_segmentation: Whether to include mask targets.

    Returns:
        Minimal :class:`albumentations.Compose` pipeline.
    """
    if channels == 1:
        mean, std = [0.5], [0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transforms: list[A.BasicTransform] = [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
        ToTensorV2(),
    ]

    kwargs: dict[str, Any] = {}
    if is_segmentation:
        kwargs["additional_targets"] = {"mask": "mask"}

    return A.Compose(transforms, **kwargs)
