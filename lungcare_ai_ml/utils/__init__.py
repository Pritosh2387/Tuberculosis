"""
Public API surface for the LungCare AI ``utils`` package.

Imports are grouped by dependency weight:
- Lightweight (logger, seed, config) are exported unconditionally.
- Heavier modules (checkpoint, dicom_utils, visualization) expose their
  primary classes / functions here but can also be imported directly from
  their own modules to avoid loading unnecessary heavy dependencies (e.g.
  importing only ``get_logger`` without pulling in ``torch`` or ``cv2``).
"""

from utils.checkpoint import CheckpointEntry, CheckpointManager
from utils.config import (
    AugmentationConfig,
    AugmentationSplitConfig,
    BaseConfig,
    CheckpointingConfig,
    ClassificationConfig,
    ConfigLoader,
    DeviceConfig,
    EarlyStoppingConfig,
    InferenceConfig,
    InputConfig,
    LossConfig,
    LoggingConfig,
    PathsConfig,
    ProjectConfig,
    ReproducibilityConfig,
    SchedulerConfig,
    SegmentationConfig,
    TransformConfig,
    WandbConfig,
)
from utils.dicom_utils import (
    DicomMetadata,
    WindowPreset,
    apply_windowing,
    dicom_to_image,
    get_window_preset,
    load_dicom,
    load_dicom_series,
    normalize_pixel_array,
)
from utils.logger import get_logger, reset_logging, setup_logging
from utils.seed import (
    SeedState,
    get_seed_state,
    restore_seed_state,
    set_seed,
    worker_init_fn,
)
from utils.visualization import (
    apply_colormap,
    create_prediction_card,
    overlay_heatmap,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_training_history,
    save_heatmap,
    save_overlay,
    visualize_segmentation,
)

__all__ = [
    # logger
    "get_logger",
    "setup_logging",
    "reset_logging",
    # seed
    "SeedState",
    "set_seed",
    "get_seed_state",
    "restore_seed_state",
    "worker_init_fn",
    # config models
    "ConfigLoader",
    "BaseConfig",
    "ProjectConfig",
    "PathsConfig",
    "DeviceConfig",
    "ReproducibilityConfig",
    "LoggingConfig",
    "WandbConfig",
    "ClassificationConfig",
    "SegmentationConfig",
    "AugmentationConfig",
    "AugmentationSplitConfig",
    "TransformConfig",
    "InferenceConfig",
    "InputConfig",
    "LossConfig",
    "EarlyStoppingConfig",
    "SchedulerConfig",
    "CheckpointingConfig",
    # checkpoint
    "CheckpointManager",
    "CheckpointEntry",
    # dicom
    "DicomMetadata",
    "WindowPreset",
    "load_dicom",
    "apply_windowing",
    "get_window_preset",
    "normalize_pixel_array",
    "dicom_to_image",
    "load_dicom_series",
    # visualization
    "apply_colormap",
    "overlay_heatmap",
    "save_heatmap",
    "save_overlay",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_training_history",
    "visualize_segmentation",
    "create_prediction_card",
]
