"""
models/densenet.py
───────────────────
DenseNet121 classifier — the backbone of CheXNet (Stanford, 2017),
which demonstrated radiologist-level chest X-ray performance.

Key difference from ResNet: dense connectivity concatenates feature maps
from ALL preceding layers, preserving both low-level texture features
and high-level semantic features simultaneously.

Grad-CAM target: features.denseblock4 — the final dense block.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights

logger = logging.getLogger("lungcare.models.densenet")


class DenseNet121Classifier(nn.Module):
    """
    DenseNet121-based chest X-ray classifier.

    Args:
        num_classes:  Number of output logits.
        pretrained:   Load ImageNet weights.
        dropout_rate: Dropout before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Keep feature extractor; replace classifier
        self.features = backbone.features       # conv + denseblocks
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, num_classes),
        )
        logger.info("DenseNet121Classifier | num_classes=%d | pretrained=%s",
                    num_classes, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B, num_classes)."""
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 1024, 7, 7) spatial feature maps after denseblock4."""
        return self.features(x)

    def get_target_layer(self) -> nn.Module:
        """Return denseblock4 as the Grad-CAM hook target."""
        return self.features.denseblock4
