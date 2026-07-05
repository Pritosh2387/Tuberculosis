"""
DenseNet121 classifier for LungCare AI.

Uses a pretrained DenseNet121 backbone from ``torchvision``.  The original
``classifier`` (Linear) is replaced by a custom head.  The full ``features``
Sequential is kept intact so ``features.denseblock4`` is directly accessible
as the Grad-CAM target.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

from models.classification.base_classifier import BaseClassifier

logger = logging.getLogger("lungcare.models.densenet")

_FEATURE_DIM: int = 1024


class DenseNet121Classifier(BaseClassifier):
    """
    DenseNet121-based chest X-ray classifier.

    The standard ``classifier`` Linear layer is replaced with::

        ReLU → AdaptiveAvgPool2d(1,1) → Flatten → Dropout → Linear(1024, num_classes)

    The ReLU and pooling are absorbed into ``self.classifier`` so that
    ``self.features`` outputs raw ``norm5`` activations — the tensor that
    Grad-CAM reads as spatial feature maps.

    Grad-CAM target: ``self.features.denseblock4``.

    Args:
        num_classes: Number of output logits.
        task: ``'binary'``, ``'multiclass'``, or ``'multilabel'``.
        pretrained: Load ImageNet-pretrained weights.
        freeze_backbone: Freeze backbone weights after construction.
        dropout_rate: Dropout probability in the classification head.
    """

    def __init__(
        self,
        num_classes: int = 6,
        task: str = "multiclass",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__(num_classes=num_classes, task=task, dropout_rate=dropout_rate)

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)

        self.features = backbone.features

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(_FEATURE_DIM, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        logger.info("DenseNet121Classifier | classes=%d | task=%s | pretrained=%s",
                    num_classes, task, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes)``."""
        feat = self.features(x)
        return self.classifier(feat)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return ``norm5`` output (before ReLU and global pooling).

        Shape: ``(B, 1024, 7, 7)`` for 224×224 input.
        This tensor is used by the CAM / Grad-CAM modules as the spatial
        activation map.
        """
        return self.features(x)

    def get_target_layer(self) -> nn.Module:
        """Return ``denseblock4`` — the Grad-CAM target layer."""
        return self.features.denseblock4  # type: ignore[attr-defined]
