"""
datasets/transforms.py
───────────────────────
Albumentations transform pipelines for LungCare AI.

``build_transforms(split, image_size)``
    Returns a classification transform pipeline (image only).

``build_seg_transforms(split, image_size)``
    Returns a segmentation pipeline that applies the SAME geometric
    transforms to both image and mask simultaneously.

Why Albumentations instead of torchvision.transforms?
- Supports joint image+mask augmentation via ``additional_targets``
- 5-10x faster than PIL-based transforms for large datasets
- Rich augmentation library (noise, grid distortion, elastic, etc.)
"""
from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics — used because all backbones are pretrained on ImageNet.
# Applying these even to X-rays preserves the input distribution the
# pretrained backbone expects.
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def build_transforms(split: str, image_size: int = 224) -> A.Compose:
    """
    Build a classification transform pipeline.

    Train split: mild geometric + colour augmentation.
    Val/test:    deterministic resize + normalize only.

    Args:
        split:      ``'train'``, ``'val'``, or ``'test'``.
        image_size: Square resize target in pixels.

    Returns:
        ``albumentations.Compose`` pipeline.
    """
    resize    = A.Resize(height=image_size, width=image_size)
    normalize = A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0)
    to_tensor = ToTensorV2()

    if split == "train":
        return A.Compose([
            resize,
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            normalize,
            to_tensor,
        ])

    # val / test — deterministic only
    return A.Compose([resize, normalize, to_tensor])


def build_seg_transforms(split: str, image_size: int = 224) -> A.Compose:
    """
    Build a segmentation transform pipeline.

    Albumentations applies the SAME random augmentation to both the
    image and the mask when ``additional_targets={'mask': 'mask'}`` is
    set.  This is essential: a flipped image with an un-flipped mask
    produces meaningless training data.

    Args:
        split:      ``'train'``, ``'val'``, or ``'test'``.
        image_size: Square resize target in pixels.

    Returns:
        ``albumentations.Compose`` with ``additional_targets`` set.
    """
    resize    = A.Resize(height=image_size, width=image_size)
    normalize = A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0)
    to_tensor = ToTensorV2()

    if split == "train":
        pipeline = [
            resize,
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=10, p=0.3),
            normalize,
            to_tensor,
        ]
    else:
        pipeline = [resize, normalize, to_tensor]

    return A.Compose(pipeline, additional_targets={"mask": "mask"})
