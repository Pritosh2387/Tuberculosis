"""
Pydantic v2 configuration models and YAML loader for LungCare AI.

Every YAML config file has a matching Pydantic model that validates types,
applies defaults, and documents allowed values.  The :class:`ConfigLoader`
class is the single entry point for consuming config files.

Dependency policy:  This module imports **only** third-party libraries
(``pydantic``, ``pyyaml``) and the standard library.  It must never import
from other ``utils`` sub-modules to prevent circular imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# ─── Shared primitives ────────────────────────────────────────────────────────


class ProjectConfig(BaseModel):
    """Top-level project identity fields."""

    name: str = "LungCare AI"
    version: str = "1.0.0"
    description: str = ""


class PathsConfig(BaseModel):
    """All filesystem paths used by the project (relative to project root)."""

    data_root: str = "data"
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    cache_dir: str = "data/cache"
    splits_dir: str = "data/splits"
    checkpoints_dir: str = "checkpoints"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    reports_dir: str = "outputs/reports"
    heatmaps_dir: str = "outputs/heatmaps"
    masks_dir: str = "outputs/masks"
    overlays_dir: str = "outputs/overlays"
    healthy_references_dir: str = "data/healthy_references"

    def resolve(self, root: Path) -> "PathsConfig":
        """
        Return a copy of this config with all paths resolved under *root*.

        Args:
            root: Absolute path to the project root directory.

        Returns:
            A new :class:`PathsConfig` with absolute string paths.
        """
        data = self.model_dump()
        return PathsConfig(**{k: str(root / v) for k, v in data.items()})

    def make_dirs(self, root: Path) -> None:
        """Create all configured directories under *root* (idempotent)."""
        for rel_path in self.model_dump().values():
            (root / rel_path).mkdir(parents=True, exist_ok=True)


class DeviceConfig(BaseModel):
    """Hardware acceleration and DataLoader concurrency settings."""

    use_cuda: bool = True
    cuda_device: int = 0
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


class ReproducibilityConfig(BaseModel):
    """Random seed and algorithmic determinism settings."""

    seed: int = 42
    deterministic: bool = False


class LoggingConfig(BaseModel):
    """Python logging configuration."""

    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/lungcare_ai.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5
    log_format: str = (
        "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
    )
    datefmt: str = "%Y-%m-%d %H:%M:%S"


class WandbConfig(BaseModel):
    """Weights & Biases experiment tracking settings."""

    enabled: bool = False
    project: str = "lungcare-ai"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    mode: Literal["online", "offline", "disabled"] = "online"


class TensorboardConfig(BaseModel):
    """TensorBoard logging settings."""

    enabled: bool = True
    log_dir: str = "logs/tensorboard"
    flush_secs: int = 30


class BaseConfig(BaseModel):
    """Root configuration loaded from ``configs/base_config.yaml``."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    reproducibility: ReproducibilityConfig = Field(
        default_factory=ReproducibilityConfig
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    tensorboard: TensorboardConfig = Field(default_factory=TensorboardConfig)


# ─── Input / Augmentation shared ─────────────────────────────────────────────


class InputConfig(BaseModel):
    """Image input shape and normalisation statistics."""

    image_size: int = 224
    channels: int = 3
    normalize_mean: list[float] = Field(
        default_factory=lambda: [0.485, 0.456, 0.406]
    )
    normalize_std: list[float] = Field(
        default_factory=lambda: [0.229, 0.224, 0.225]
    )


class TransformConfig(BaseModel):
    """A single Albumentations transform with its constructor parameters."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class AugmentationSplitConfig(BaseModel):
    """Transform pipelines for each data split."""

    train: list[TransformConfig] = Field(default_factory=list)
    val: list[TransformConfig] = Field(default_factory=list)
    test: list[TransformConfig] = Field(default_factory=list)


class AugmentationConfig(BaseModel):
    """Root augmentation config loaded from ``configs/augmentation_config.yaml``."""

    classification: AugmentationSplitConfig = Field(
        default_factory=AugmentationSplitConfig
    )
    segmentation: AugmentationSplitConfig = Field(
        default_factory=AugmentationSplitConfig
    )


# ─── Classification config ────────────────────────────────────────────────────


class ArchitectureDetails(BaseModel):
    """Per-architecture metadata used by explainability modules."""

    target_layer: str = ""
    feature_dim: int = 1024
    patch_size: int | None = None
    img_size: int | None = None


class ClassificationModelConfig(BaseModel):
    """Model architecture and head settings for classification."""

    architecture: str = "densenet121"
    num_classes: int = 6
    task: Literal["binary", "multiclass", "multilabel"] = "multiclass"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.3
    label_smoothing: float = 0.1
    architectures: dict[str, ArchitectureDetails] = Field(default_factory=dict)

    def get_target_layer(self) -> str:
        """Return the Grad-CAM target layer for the selected architecture."""
        details = self.architectures.get(self.architecture)
        return details.target_layer if details else ""


class EarlyStoppingConfig(BaseModel):
    """Early-stopping callback settings."""

    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"


class SchedulerConfig(BaseModel):
    """Learning-rate scheduler settings (shared by classification & segmentation)."""

    name: str = "cosine_annealing"
    warmup_epochs: int = 5
    T_max: int = 50
    eta_min: float = 1e-7
    factor: float = 0.5
    patience: int = 7
    min_lr: float = 1e-7


class CheckpointingConfig(BaseModel):
    """Checkpoint selection and naming settings."""

    save_top_k: int = 3
    monitor: str = "val_f1"
    mode: Literal["min", "max"] = "max"
    save_last: bool = True
    filename_template: str = "{epoch:03d}-{val_f1:.4f}"


class ClassificationTrainingConfig(BaseModel):
    """Optimiser, scheduler, and callback settings for classifier training."""

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    accumulation_steps: int = 1
    val_check_interval: int = 1
    log_every_n_steps: int = 10
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=EarlyStoppingConfig
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    checkpointing: CheckpointingConfig = Field(
        default_factory=CheckpointingConfig
    )


class DatasetSourceConfig(BaseModel):
    """Common fields shared by every dataset source."""

    enabled: bool = True
    root: str = ""


class NihChestXray14Config(DatasetSourceConfig):
    """NIH ChestX-ray14 dataset paths."""

    label_file: str = "Data_Entry_2017.csv"
    image_dir: str = "images"


class MontgomeryConfig(DatasetSourceConfig):
    """Montgomery County TB dataset paths."""

    abnormal_dir: str = "CXR_png/abnormal"
    normal_dir: str = "CXR_png/normal"


class ShenzhenConfig(DatasetSourceConfig):
    """Shenzhen TB dataset paths."""

    abnormal_dir: str = "CXR_png/abnormal"
    normal_dir: str = "CXR_png/normal"


class RsnaPneumoniaConfig(DatasetSourceConfig):
    """RSNA Pneumonia Detection Challenge dataset paths."""

    label_file: str = "stage_2_train_labels.csv"
    image_dir: str = "stage_2_train_images"


class CovidQuExConfig(DatasetSourceConfig):
    """COVID-QU-Ex dataset paths with pre-defined split directories."""

    split_dirs: dict[str, str] = Field(
        default_factory=lambda: {"train": "train", "val": "val", "test": "test"}
    )


class ClassificationDatasetsConfig(BaseModel):
    """All classification dataset source configs."""

    nih_chestxray14: NihChestXray14Config = Field(
        default_factory=NihChestXray14Config
    )
    montgomery: MontgomeryConfig = Field(default_factory=MontgomeryConfig)
    shenzhen: ShenzhenConfig = Field(default_factory=ShenzhenConfig)
    rsna_pneumonia: RsnaPneumoniaConfig = Field(
        default_factory=RsnaPneumoniaConfig
    )
    covid_qu_ex: CovidQuExConfig = Field(default_factory=CovidQuExConfig)


class ClassificationDataConfig(BaseModel):
    """Split ratios, balancing strategy, and dataset sources for classification."""

    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    class_balancing: str = "weighted_sampler"
    cache_dataset: bool = False
    cache_format: str = "pt"
    datasets: ClassificationDatasetsConfig = Field(
        default_factory=ClassificationDatasetsConfig
    )

    @model_validator(mode="after")
    def _splits_must_sum_to_one(self) -> "ClassificationDataConfig":
        total = self.train_split + self.val_split + self.test_split
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0, got {total:.3f}."
            )
        return self


class ClassificationConfig(BaseModel):
    """Root classification config loaded from ``configs/classification_config.yaml``."""

    model: ClassificationModelConfig = Field(
        default_factory=ClassificationModelConfig
    )
    classes: list[str] = Field(
        default_factory=lambda: [
            "Healthy",
            "Tuberculosis",
            "Pneumonia",
            "COVID-19",
            "Lung Cancer",
            "Pulmonary Fibrosis",
        ]
    )
    class_weights: dict[str, float] = Field(default_factory=dict)
    input: InputConfig = Field(default_factory=InputConfig)
    training: ClassificationTrainingConfig = Field(
        default_factory=ClassificationTrainingConfig
    )
    data: ClassificationDataConfig = Field(
        default_factory=ClassificationDataConfig
    )


# ─── Segmentation config ──────────────────────────────────────────────────────


class SegmentationModelConfig(BaseModel):
    """Model architecture settings for segmentation."""

    architecture: str = "attention_unet"
    in_channels: int = 1
    out_channels: int = 1
    features: list[int] = Field(default_factory=lambda: [64, 128, 256, 512])
    bilinear: bool = True
    dropout_rate: float = 0.2


class LossConfig(BaseModel):
    """Combined loss function weights and hyperparameters."""

    name: str = "combined"
    dice_weight: float = 0.5
    bce_weight: float = 0.3
    focal_weight: float = 0.2
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    smooth: float = 1e-6

    @model_validator(mode="after")
    def _weights_must_sum_to_one(self) -> "LossConfig":
        if self.name == "combined":
            total = self.dice_weight + self.bce_weight + self.focal_weight
            if not (0.999 <= total <= 1.001):
                raise ValueError(
                    f"dice_weight + bce_weight + focal_weight must equal 1.0, got {total:.3f}."
                )
        return self


class SegmentationInputConfig(BaseModel):
    """Image input shape for segmentation (typically single-channel)."""

    image_size: int = 256
    channels: int = 1
    normalize_mean: list[float] = Field(default_factory=lambda: [0.5])
    normalize_std: list[float] = Field(default_factory=lambda: [0.5])


class SiimAcrConfig(DatasetSourceConfig):
    """SIIM-ACR Pneumothorax dataset paths."""

    image_dir: str = "dicom-images-train"
    mask_dir: str = "masks"
    label_file: str = "train-rle.csv"


class MontgomeryMasksConfig(DatasetSourceConfig):
    """Montgomery County lung mask dataset paths."""

    image_dir: str = "CXR_png"
    left_mask_dir: str = "ManualMask/leftMask"
    right_mask_dir: str = "ManualMask/rightMask"


class CovidQuExSegConfig(DatasetSourceConfig):
    """COVID-QU-Ex segmentation mask dataset paths."""

    image_dir: str = "images"
    mask_dir: str = "infection_masks"


class MosMedDataConfig(DatasetSourceConfig):
    """MosMedData CT segmentation dataset paths."""

    ct_dir: str = "studies"
    mask_dir: str = "masks"


class LidcIdriConfig(DatasetSourceConfig):
    """LIDC-IDRI nodule detection dataset paths."""

    nodule_size_threshold: int = 3


class SegmentationDatasetsConfig(BaseModel):
    """All segmentation dataset source configs."""

    siim_acr_pneumothorax: SiimAcrConfig = Field(
        default_factory=SiimAcrConfig
    )
    montgomery_masks: MontgomeryMasksConfig = Field(
        default_factory=MontgomeryMasksConfig
    )
    covid_qu_ex_seg: CovidQuExSegConfig = Field(
        default_factory=CovidQuExSegConfig
    )
    mosmeddata: MosMedDataConfig = Field(default_factory=MosMedDataConfig)
    lidc_idri: LidcIdriConfig = Field(default_factory=LidcIdriConfig)


class SegmentationDataConfig(BaseModel):
    """Split ratios and dataset sources for segmentation."""

    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    class_balancing: str = "none"
    cache_dataset: bool = False
    datasets: SegmentationDatasetsConfig = Field(
        default_factory=SegmentationDatasetsConfig
    )

    @model_validator(mode="after")
    def _splits_must_sum_to_one(self) -> "SegmentationDataConfig":
        total = self.train_split + self.val_split + self.test_split
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"train_split + val_split + test_split must equal 1.0, got {total:.3f}."
            )
        return self


class SegmentationTrainingConfig(BaseModel):
    """Optimiser, scheduler, and callback settings for segmentation training."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    mixed_precision: bool = True
    gradient_clip_val: float = 1.0
    accumulation_steps: int = 2
    val_check_interval: int = 1
    log_every_n_steps: int = 20
    early_stopping: EarlyStoppingConfig = Field(
        default_factory=lambda: EarlyStoppingConfig(
            patience=15, monitor="val_dice", mode="max"
        )
    )
    scheduler: SchedulerConfig = Field(
        default_factory=lambda: SchedulerConfig(
            name="reduce_on_plateau", factor=0.5, patience=7
        )
    )
    checkpointing: CheckpointingConfig = Field(
        default_factory=lambda: CheckpointingConfig(
            monitor="val_dice",
            mode="max",
            filename_template="{epoch:03d}-{val_dice:.4f}",
        )
    )


class SegmentationConfig(BaseModel):
    """Root segmentation config loaded from ``configs/segmentation_config.yaml``."""

    model: SegmentationModelConfig = Field(
        default_factory=SegmentationModelConfig
    )
    input: SegmentationInputConfig = Field(
        default_factory=SegmentationInputConfig
    )
    loss: LossConfig = Field(default_factory=LossConfig)
    metrics: list[str] = Field(
        default_factory=lambda: [
            "dice",
            "iou",
            "precision",
            "recall",
            "sensitivity",
            "specificity",
        ]
    )
    training: SegmentationTrainingConfig = Field(
        default_factory=SegmentationTrainingConfig
    )
    data: SegmentationDataConfig = Field(default_factory=SegmentationDataConfig)


# ─── Inference config ─────────────────────────────────────────────────────────


class InferenceClassificationModelConfig(BaseModel):
    """Classifier model settings for inference."""

    checkpoint_path: str | None = None
    architecture: str = "densenet121"
    num_classes: int = 6
    task: str = "multiclass"


class InferenceSegmentationModelConfig(BaseModel):
    """Segmentation model settings for inference."""

    checkpoint_path: str | None = None
    architecture: str = "attention_unet"
    in_channels: int = 1
    out_channels: int = 1


class InferenceModelConfig(BaseModel):
    """Combined model settings for the inference pipeline."""

    classification: InferenceClassificationModelConfig = Field(
        default_factory=InferenceClassificationModelConfig
    )
    segmentation: InferenceSegmentationModelConfig = Field(
        default_factory=InferenceSegmentationModelConfig
    )
    device: str = "auto"


class ExplainabilityConfig(BaseModel):
    """Grad-CAM / attention-map explainability settings."""

    method: str = "gradcam_plus_plus"
    target_layer: str = "features.denseblock4"
    alpha_overlay: float = 0.4
    save_raw_heatmap: bool = True
    architecture_target_layers: dict[str, str | None] = Field(
        default_factory=dict
    )

    def resolve_target_layer(self, architecture: str) -> str | None:
        """Return the target layer string for a given architecture name."""
        return self.architecture_target_layers.get(architecture, self.target_layer)


class ThresholdConfig(BaseModel):
    """Decision thresholds for classification and region detection."""

    binary_classification: float = 0.5
    disease_confidence: float = 0.30
    abnormal_region_area_px: int = 500
    mask_binarization: float = 0.5
    opacity_severity_mild: float = 0.30
    opacity_severity_moderate: float = 0.60


class PostprocessingConfig(BaseModel):
    """Heatmap and mask postprocessing settings."""

    heatmap_colormap: str = "jet"
    morphological_cleanup: bool = True
    morph_kernel_size: int = 5
    min_contour_area: int = 200
    region_padding_px: int = 10


class InferenceOutputConfig(BaseModel):
    """Output file and format settings for inference results."""

    save_heatmap: bool = True
    save_mask: bool = True
    save_overlay: bool = True
    save_annotated: bool = True
    output_dir: str = "outputs/inference"
    report_format: Literal["json", "yaml"] = "json"


class HealthyReferenceConfig(BaseModel):
    """Healthy scan comparison settings."""

    enabled: bool = True
    reference_dir: str = "data/healthy_references"
    comparison_method: Literal["ssim", "histogram", "feature"] = "ssim"
    ssim_window_size: int = 11
    histogram_bins: int = 256


class InferenceConfig(BaseModel):
    """Root inference config loaded from ``configs/inference_config.yaml``."""

    model: InferenceModelConfig = Field(default_factory=InferenceModelConfig)
    classes: list[str] = Field(
        default_factory=lambda: [
            "Healthy",
            "Tuberculosis",
            "Pneumonia",
            "COVID-19",
            "Lung Cancer",
            "Pulmonary Fibrosis",
        ]
    )
    input: InputConfig = Field(default_factory=InputConfig)
    explainability: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig
    )
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    postprocessing: PostprocessingConfig = Field(
        default_factory=PostprocessingConfig
    )
    output: InferenceOutputConfig = Field(default_factory=InferenceOutputConfig)
    healthy_reference: HealthyReferenceConfig = Field(
        default_factory=HealthyReferenceConfig
    )


# ─── Config Loader ────────────────────────────────────────────────────────────


class ConfigLoader:
    """
    Load and validate YAML config files into typed Pydantic models.

    All paths are resolved relative to *config_dir* (typically
    ``<project_root>/configs/``).

    Args:
        config_dir: Path to the directory containing the YAML config files.

    Example::

        loader = ConfigLoader("configs")
        base = loader.load_base()
        clf = loader.load_classification()
    """

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir)
        if not self.config_dir.is_dir():
            raise NotADirectoryError(
                f"Config directory not found: {self.config_dir}"
            )

    def _read_yaml(self, filename: str) -> dict[str, Any]:
        """Load a YAML file and return its contents as a plain dict."""
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}"
            )
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively merge *override* into *base*.

        Nested dicts are merged; all other types in *override* replace *base*.

        Args:
            base: The base configuration dict.
            override: Values that take precedence over *base*.

        Returns:
            A new merged dict (neither input is mutated).
        """
        result: dict[str, Any] = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def load_base(self) -> BaseConfig:
        """Load and validate ``base_config.yaml``."""
        data = self._read_yaml("base_config.yaml")
        return BaseConfig.model_validate(data)

    def load_classification(self) -> ClassificationConfig:
        """Load and validate ``classification_config.yaml``."""
        data = self._read_yaml("classification_config.yaml")
        return ClassificationConfig.model_validate(data)

    def load_segmentation(self) -> SegmentationConfig:
        """Load and validate ``segmentation_config.yaml``."""
        data = self._read_yaml("segmentation_config.yaml")
        return SegmentationConfig.model_validate(data)

    def load_augmentation(self) -> AugmentationConfig:
        """Load and validate ``augmentation_config.yaml``."""
        data = self._read_yaml("augmentation_config.yaml")
        return AugmentationConfig.model_validate(data)

    def load_inference(self) -> InferenceConfig:
        """Load and validate ``inference_config.yaml``."""
        data = self._read_yaml("inference_config.yaml")
        return InferenceConfig.model_validate(data)

    def load_all(
        self,
    ) -> tuple[
        BaseConfig,
        ClassificationConfig,
        SegmentationConfig,
        AugmentationConfig,
        InferenceConfig,
    ]:
        """
        Load and validate all five config files in one call.

        Returns:
            A tuple of ``(base, classification, segmentation, augmentation, inference)``.
        """
        return (
            self.load_base(),
            self.load_classification(),
            self.load_segmentation(),
            self.load_augmentation(),
            self.load_inference(),
        )

    def load_with_overrides(
        self, filename: str, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Load a YAML file and apply a dict of overrides before validation.

        Useful for CLI-level hyperparameter sweeps without editing YAML files.

        Args:
            filename: YAML filename (e.g. ``'classification_config.yaml'``).
            overrides: Flat or nested dict of values to override.

        Returns:
            The merged raw dict (caller is responsible for validation).
        """
        base = self._read_yaml(filename)
        return self._deep_merge(base, overrides)
