"""
Public API surface for the LungCare AI ``datasets`` package.
"""

from datasets.base_dataset import BaseDataset
from datasets.classification_dataset import ClassificationDataset
from datasets.dicom_dataset import DicomDataset
from datasets.segmentation_dataset import SegmentationDataset, mask_to_rle, rle_to_mask
from datasets.transforms import (
    build_transform,
    get_classification_transforms,
    get_identity_transform,
    get_segmentation_transforms,
)

__all__ = [
    # Base
    "BaseDataset",
    # Concrete datasets
    "ClassificationDataset",
    "SegmentationDataset",
    "DicomDataset",
    # Transform builders
    "build_transform",
    "get_classification_transforms",
    "get_segmentation_transforms",
    "get_identity_transform",
    # RLE utilities
    "rle_to_mask",
    "mask_to_rle",
]
