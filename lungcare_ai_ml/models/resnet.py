"""
models/resnet.py
─────────────────
ResNet50 classifier for chest X-ray multi-disease classification.

The standard ResNet50 (ImageNet pretrained) backbone with a
task-specific classification head:

    conv1 → bn1 → relu → maxpool
    → layer1 → layer2 → layer3 → layer4   ← Grad-CAM target
    → GlobalAvgPool → Dropout → Linear(2048, num_classes)

Layer references (self.layer1 … self.layer4) are kept as named
attributes so GradCAM can hook into layer4 without unwrapping a
Sequential.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger("lungcare.models.resnet")


class ResNet50Classifier(nn.Module):
    """
    ResNet50-based chest X-ray classifier.

    Args:
        num_classes:   Number of output logits (default 2 for TB binary).
        pretrained:    Load ImageNet weights.
        dropout_rate:  Dropout probability before the final linear layer.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Retain individual layer references for Grad-CAM hooks
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1
        self.layer2  = backbone.layer2
        self.layer3  = backbone.layer3
        self.layer4  = backbone.layer4   # Grad-CAM target
        self.avgpool = backbone.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(2048, num_classes),
        )
        logger.info("ResNet50Classifier | num_classes=%d | pretrained=%s",
                    num_classes, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, num_classes)."""
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
        """Return (B, 2048, 7, 7) spatial feature maps from layer4."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def get_target_layer(self) -> nn.Module:
        """Return layer4 as the Grad-CAM hook target."""
        return self.layer4
