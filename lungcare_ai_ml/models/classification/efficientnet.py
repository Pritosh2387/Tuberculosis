"""
EfficientNet-B0 classifier for LungCare AI.

Uses ``timm`` to load a pretrained EfficientNet-B0 backbone.  The global
pooling and classifier head are removed (``global_pool=''``) so the backbone
outputs spatial feature maps ``(B, 1280, H, W)``.  A custom pooling +
classification head is added on top.

Grad-CAM target: ``self.backbone.conv_head`` (the 1×1 expansion conv that
produces 1280-channel spatial features immediately before global pooling).
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn

from models.classification.base_classifier import BaseClassifier

logger = logging.getLogger("lungcare.models.efficientnet")

_TIMM_MODEL: str = "efficientnet_b0"


class EfficientNetB0Classifier(BaseClassifier):
    """
    EfficientNet-B0–based chest X-ray classifier.

    Classification head::

        AdaptiveAvgPool2d(1,1) → Flatten → Dropout → Linear(1280, num_classes)

    Grad-CAM target: ``self.backbone.conv_head``.

    Args:
        num_classes: Number of output logits.
        task: ``'binary'``, ``'multiclass'``, or ``'multilabel'``.
        pretrained: Load ImageNet-pretrained weights via ``timm``.
        freeze_backbone: Freeze backbone after construction.
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

        self.backbone = timm.create_model(
            _TIMM_MODEL,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        in_features: int = self.backbone.num_features

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        logger.info(
            "EfficientNetB0Classifier | classes=%d | task=%s | "
            "in_features=%d | pretrained=%s",
            num_classes, task, in_features, pretrained,
        )

    def _extract_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return spatial feature maps ``(B, 1280, H, W)``."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, num_classes)``."""
        spatial = self._extract_spatial(x)
        pooled = self.pool(spatial).flatten(1)
        return self.classifier(pooled)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return spatial feature maps ``(B, 1280, H, W)``.

        With 224×224 input: ``H = W = 7``.
        """
        return self._extract_spatial(x)

    def get_target_layer(self) -> nn.Module:
        """
        Return ``conv_head`` — EfficientNet's 1×1 expansion conv.

        This is the last convolutional layer before global average pooling
        and produces ``(B, 1280, H, W)`` spatial activations.
        """
        return self.backbone.conv_head  # type: ignore[attr-defined]
