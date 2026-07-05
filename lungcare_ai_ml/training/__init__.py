"""
Public API for the LungCare AI ``training`` package.
"""

# Apply the torch/transformers pytree compatibility shim before any import
# that transitively loads ``transformers`` via ``torchmetrics`` (see
# :mod:`utils.torch_compat`).  Idempotent and a no-op on modern torch.
from utils.torch_compat import ensure_pytree_compat as _ensure_pytree_compat

_ensure_pytree_compat()

from training.classification_trainer import ClassificationConfig, ClassificationTrainer
from training.losses import (
    BCEDiceLoss,
    DeepSupervisionLoss,
    DiceLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    build_classification_loss,
    build_segmentation_loss,
)
from training.metrics import ClassificationMetrics, SegmentationMetrics
from training.schedulers import WarmupCosineScheduler, build_scheduler
from training.segmentation_trainer import SegmentationConfig, SegmentationTrainer
from training.trainer import (
    BaseTrainer,
    EarlyStopping,
    EarlyStoppingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainerConfig,
)

__all__ = [
    # Trainers
    "BaseTrainer",
    "ClassificationTrainer",
    "SegmentationTrainer",
    # Configs
    "TrainerConfig",
    "ClassificationConfig",
    "SegmentationConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "EarlyStoppingConfig",
    # Losses
    "FocalLoss",
    "DiceLoss",
    "BCEDiceLoss",
    "LabelSmoothingCrossEntropy",
    "DeepSupervisionLoss",
    "build_classification_loss",
    "build_segmentation_loss",
    # Metrics
    "ClassificationMetrics",
    "SegmentationMetrics",
    # Schedulers
    "WarmupCosineScheduler",
    "build_scheduler",
    # Utilities
    "EarlyStopping",
]
