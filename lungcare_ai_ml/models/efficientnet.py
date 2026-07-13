"""
models/efficientnet.py
───────────────────────
EfficientNet-B0 classifier loaded via timm.

EfficientNet uses compound scaling: width, depth, and resolution are
scaled simultaneously using a fixed ratio found by NAS. B0 achieves
better accuracy than ResNet50 with 4× fewer parameters.

loaded via timm's create_model() with num_classes=0 to remove the
default head, then a custom classification head is attached.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("lungcare.models.efficientnet")

_FEATURE_DIM = 1280   # EfficientNet-B0 last-stage channels


class EfficientNetB0Classifier(nn.Module):
    """
    EfficientNet-B0 chest X-ray classifier.

    Args:
        num_classes:  Number of output logits.
        pretrained:   Load ImageNet weights from timm.
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

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for EfficientNetB0Classifier. "
                "Install with: pip install timm"
            ) from e

        # num_classes=0 removes the default classification head
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,          # returns features only
            global_pool="avg",      # includes global avg pool
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(_FEATURE_DIM, num_classes),
        )
        logger.info("EfficientNetB0Classifier | num_classes=%d | pretrained=%s",
                    num_classes, pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B, num_classes)."""
        features = self.backbone(x)    # (B, 1280)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return spatial feature maps before global pooling.
        Used by GradCAM.
        """
        return self.backbone.forward_features(x)

    def get_target_layer(self) -> nn.Module:
        """Return the last conv block as Grad-CAM target."""
        return self.backbone.conv_head
