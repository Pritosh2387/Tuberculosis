"""
models/__init__.py
───────────────────
Model factory for LungCare AI.

Single entry point: ``create_model(name, **kwargs)`` returns a
classifier. Imports are lazy (inside the factory function) to avoid
the broken torchvision→torch.onnx→transformers chain at collection time.

Registered classifiers
-----------------------
  resnet50         → ResNet50Classifier
  densenet121      → DenseNet121Classifier
  efficientnet_b0  → EfficientNetB0Classifier
  vit_b16          → ViTClassifier
"""
from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, str] = {
    "resnet50":        "models.resnet.ResNet50Classifier",
    "densenet121":     "models.densenet.DenseNet121Classifier",
    "efficientnet_b0": "models.efficientnet.EfficientNetB0Classifier",
    "vit_b16":         "models.vit.ViTClassifier",
}


def create_model(name: str, **kwargs: Any) -> Any:
    """
    Instantiate a classifier by name (lazy import — models loaded on demand).

    Args:
        name:     Architecture key. One of: resnet50, densenet121,
                  efficientnet_b0, vit_b16.
        **kwargs: Forwarded to the model constructor
                  (num_classes, pretrained, dropout_rate, …).

    Returns:
        Instantiated model with forward(), get_features(), get_target_layer().

    Raises:
        KeyError: Unknown architecture name.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown architecture '{name}'. "
            f"Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls    = getattr(module, class_name)
    return cls(**kwargs)


__all__ = ["create_model"]
