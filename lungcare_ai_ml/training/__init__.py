"""training/__init__.py"""
from training.losses import (
    BCEDiceLoss,
    DiceLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    build_classification_loss,
    build_segmentation_loss,
)
from training.trainer import EarlyStopping, Trainer, load_checkpoint, save_checkpoint

# ClassificationMetrics and SegmentationMetrics are imported lazily in
# their respective modules to avoid the torchmetrics→transformers chain
# at collection time. Import them directly if needed:
#   from training.metrics import ClassificationMetrics, SegmentationMetrics

__all__ = [
    "Trainer",
    "EarlyStopping",
    "save_checkpoint",
    "load_checkpoint",
    "FocalLoss",
    "DiceLoss",
    "BCEDiceLoss",
    "LabelSmoothingCrossEntropy",
    "build_classification_loss",
    "build_segmentation_loss",
]
