"""
ResNet50 classifier for LungCare AI.

Uses a pretrained ResNet50 backbone from ``torchvision`` with the fully
connected layer replaced by a configurable classification head.  All
four residual stages are kept as individually addressable attributes so
that :class:`GradCAM` can hook into ``layer4`` without traversing a
nested Sequential.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

from models.classification.base_classifier import BaseClassifier

logger = logging.getLogger("lungcare.models.resnet")

_FEATURE_DIM: int = 2048


class ResNet50Classifier(BaseClassifier):
    """
    ResNet50-based chest X-ray classifier.

    The standard ResNet50 ``fc`` layer is replaced with::

        Dropout(dropout_rate) → Linear(2048, num_classes)

    Grad-CAM target: ``self.layer4`` (spatial output before GlobalAvgPool).

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

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Retain individual layer references for clean Grad-CAM targeting
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4      # Grad-CAM target
        self.avgpool = backbone.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(_FEATURE_DIM, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        logger.info("ResNet50Classifier | classes=%d | task=%s | pretrained=%s",
                    num_classes, task, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes)``."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return spatial feature maps from ``layer4``.

        Shape: ``(B, 2048, 7, 7)`` for 224×224 input.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def get_target_layer(self) -> nn.Module:
        """Return ``layer4`` — the Grad-CAM target layer."""
        return self.layer4
