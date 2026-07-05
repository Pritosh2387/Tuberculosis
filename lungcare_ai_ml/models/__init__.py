"""
Public API for the LungCare AI ``models`` package.

Provides a :func:`create_classifier` factory and re-exports the most
commonly used symbols from each sub-package.
"""

from __future__ import annotations

from typing import Any

from models.classification import (
    BaseClassifier,
    DenseNet121Classifier,
    EfficientNetB0Classifier,
    ResNet50Classifier,
    ViTClassifier,
)
from models.explainability import (
    CAM,
    AttentionHeatmap,
    AttentionRollout,
    GradCAM,
    GradCAMPlusPlus,
)
from models.segmentation import AttentionUNet, UNet, UNetPlusPlus

_CLASSIFIER_REGISTRY: dict[str, type[BaseClassifier]] = {
    "resnet50": ResNet50Classifier,
    "densenet121": DenseNet121Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
    "vit_b16": ViTClassifier,
}

_SEGMENTATION_REGISTRY: dict[str, type] = {
    "unet": UNet,
    "attention_unet": AttentionUNet,
    "unet_plus_plus": UNetPlusPlus,
}


def create_classifier(name: str, **kwargs: Any) -> BaseClassifier:
    """
    Instantiate a classifier by name.

    Args:
        name: Architecture name.  One of ``'resnet50'``, ``'densenet121'``,
            ``'efficientnet_b0'``, ``'vit_b16'``.
        **kwargs: Forwarded to the constructor.

    Returns:
        A :class:`BaseClassifier` instance.

    Raises:
        KeyError: If *name* is not registered.

    Example::

        model = create_classifier("resnet50", num_classes=6, pretrained=True)
    """
    if name not in _CLASSIFIER_REGISTRY:
        raise KeyError(
            f"Unknown classifier '{name}'. "
            f"Available: {list(_CLASSIFIER_REGISTRY)}"
        )
    return _CLASSIFIER_REGISTRY[name](**kwargs)


def create_segmentation_model(name: str, **kwargs: Any) -> Any:
    """
    Instantiate a segmentation model by name.

    Args:
        name: Architecture name.  One of ``'unet'``, ``'attention_unet'``,
            ``'unet_plus_plus'``.
        **kwargs: Forwarded to the constructor.

    Returns:
        A segmentation model instance.
    """
    if name not in _SEGMENTATION_REGISTRY:
        raise KeyError(
            f"Unknown segmentation model '{name}'. "
            f"Available: {list(_SEGMENTATION_REGISTRY)}"
        )
    return _SEGMENTATION_REGISTRY[name](**kwargs)


def get_explainability(
    method: str,
    model: BaseClassifier,
    **kwargs: Any,
) -> Any:
    """
    Instantiate an explainability method by name.

    Args:
        method: One of ``'cam'``, ``'gradcam'``, ``'gradcam++'``,
            ``'rollout'``, ``'attn_heatmap'``.
        model: A classifier instance.
        **kwargs: Forwarded to the explainability class.

    Returns:
        An explainability object.
    """
    registry: dict[str, type] = {
        "cam": CAM,
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "rollout": AttentionRollout,
        "attn_heatmap": AttentionHeatmap,
    }
    if method not in registry:
        raise KeyError(f"Unknown method '{method}'. Available: {list(registry)}")
    return registry[method](model, **kwargs)


__all__ = [
    # Classifiers
    "BaseClassifier",
    "ResNet50Classifier",
    "DenseNet121Classifier",
    "EfficientNetB0Classifier",
    "ViTClassifier",
    # Segmentation
    "UNet",
    "AttentionUNet",
    "UNetPlusPlus",
    # Explainability
    "CAM",
    "GradCAM",
    "GradCAMPlusPlus",
    "AttentionRollout",
    "AttentionHeatmap",
    # Factories
    "create_classifier",
    "create_segmentation_model",
    "get_explainability",
]
