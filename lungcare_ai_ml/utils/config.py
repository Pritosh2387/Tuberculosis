"""
utils/config.py
────────────────
Lightweight YAML configuration loader for LungCare AI.

Replaces the previous 789-line Pydantic config module with plain
Python dataclasses.  No external validation library required.

Usage
-----
    from utils.config import load_config
    cfg = load_config("config.yaml")
    print(cfg.training.lr)          # 0.0001
    print(cfg.data.num_classes)     # 2
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("lungcare.config")


@dataclass
class ProjectConfig:
    name: str = "LungCare AI"
    version: str = "2.0.0"
    seed: int = 42


@dataclass
class DataConfig:
    montgomery_dir: str = "data/montgomery"
    shenzhen_dir:   str = "data/shenzhen"
    train_csv:  str = "data/splits/train.csv"
    val_csv:    str = "data/splits/val.csv"
    test_csv:   str = "data/splits/test.csv"
    class_names: list[str] = field(
        default_factory=lambda: ["Normal", "Tuberculosis"]
    )
    num_classes: int = 2
    image_size:  int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    architecture: str = "resnet50"   # resnet50 | densenet121 | efficientnet_b0 | vit_b16
    pretrained:   bool = True
    dropout_rate: float = 0.3


@dataclass
class TrainingConfig:
    epochs:       int   = 50
    batch_size:   int   = 32
    lr:           float = 1e-4
    weight_decay: float = 1e-2
    grad_clip:    float = 1.0
    amp:          bool  = True
    scheduler:    str   = "cosine"     # cosine | step | plateau
    patience:     int   = 10
    monitor:      str   = "val_f1"
    monitor_mode: str   = "max"
    checkpoint_dir: str = "checkpoints"
    log_dir:        str = "logs"


@dataclass
class LossConfig:
    type:            str   = "cross_entropy"  # cross_entropy | focal | label_smoothing
    label_smoothing: float = 0.1
    focal_alpha:     float = 0.25
    focal_gamma:     float = 2.0


@dataclass
class InferenceConfig:
    checkpoint_path:  str        = "checkpoints/best.pth"
    device:           str        = "cpu"
    explainability:   str        = "gradcam"   # gradcam | rollout | none
    run_segmentation: bool       = False
    seg_checkpoint:   str | None = None
    threshold:        float      = 0.5
    top_k:            int        = 3
    api_host:         str        = "0.0.0.0"
    api_port:         int        = 8000


@dataclass
class Config:
    """Root configuration — one object holds everything."""
    project:   ProjectConfig   = field(default_factory=ProjectConfig)
    data:      DataConfig      = field(default_factory=DataConfig)
    model:     ModelConfig     = field(default_factory=ModelConfig)
    training:  TrainingConfig  = field(default_factory=TrainingConfig)
    loss:      LossConfig      = field(default_factory=LossConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


# ─── YAML loader ──────────────────────────────────────────────────────────────


def _populate(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate a dataclass from a dict, ignoring unknown keys."""
    if not dataclasses.is_dataclass(cls):
        return data
    valid_fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key not in valid_fields:
            continue
        field_type = valid_fields[key].type
        # Resolve string annotations
        if isinstance(field_type, str):
            field_type = eval(field_type, {"list": list, "str": str,  # noqa: S307
                                           "int": int, "float": float,
                                           "bool": bool, "None": type(None)})
        if isinstance(value, dict) and dataclasses.is_dataclass(field_type):
            kwargs[key] = _populate(field_type, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(yaml_path: str | Path = "config.yaml") -> Config:
    """
    Load ``config.yaml`` and return a :class:`Config` object.

    Missing keys fall back to dataclass defaults.
    Unknown YAML keys are silently ignored.

    Args:
        yaml_path: Path to the YAML file (default: ``config.yaml``).

    Returns:
        A fully-populated :class:`Config` instance.
    """
    path = Path(yaml_path)
    if not path.exists():
        logger.warning("Config file '%s' not found — using defaults.", path)
        return Config()

    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = Config(
        project=_populate(ProjectConfig,   raw.get("project",   {})),
        data=_populate(DataConfig,         raw.get("data",      {})),
        model=_populate(ModelConfig,       raw.get("model",     {})),
        training=_populate(TrainingConfig, raw.get("training",  {})),
        loss=_populate(LossConfig,         raw.get("loss",      {})),
        inference=_populate(InferenceConfig, raw.get("inference", {})),
    )
    logger.info(
        "Config loaded | model=%s | num_classes=%d | epochs=%d",
        cfg.model.architecture, cfg.data.num_classes, cfg.training.epochs,
    )
    return cfg
