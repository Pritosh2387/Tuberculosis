"""
models/vit.py
──────────────
Vision Transformer (ViT-B/16) classifier loaded via timm.

ViT divides the 224×224 image into 16×16 patches (→ 196 tokens),
adds a learnable [CLS] token, and uses 12 Transformer encoder layers
with multi-head self-attention. The [CLS] token output is used for
classification.

Key difference from CNNs: self-attention is GLOBAL — every patch
attends to every other patch in a single layer, enabling long-range
dependency modelling across the entire image.

Explainability: Grad-CAM does NOT work on ViT (no spatial feature
maps). Use AttentionRollout from models/gradcam.py instead.
"""
from __future__ import annotations
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("lungcare.models.vit")


class ViTClassifier(nn.Module):
    """
    ViT-B/16 chest X-ray classifier.

    Args:
        num_classes:  Number of output logits.
        pretrained:   Load ImageNet-21k weights from timm.
        dropout_rate: Dropout before the final linear layer.
        image_size:   Input resolution (must be divisible by patch_size=16).
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_rate: float = 0.1,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for ViTClassifier. "
                "Install with: pip install timm"
            ) from e

        # num_classes=0 → returns raw CLS token embedding (768-dim)
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size,
        )
        embed_dim = self.backbone.embed_dim   # 768 for ViT-B
        self.head_drop = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)
        logger.info("ViTClassifier | num_classes=%d | pretrained=%s | embed_dim=%d",
                    num_classes, pretrained, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B, num_classes)."""
        cls_token = self.backbone(x)       # (B, embed_dim)
        return self.classifier(self.head_drop(cls_token))

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return CLS token embedding. Not used by GradCAM (use AttentionRollout)."""
        return self.backbone(x)

    def get_target_layer(self) -> None:  # type: ignore[override]
        """ViT uses AttentionRollout, not GradCAM. Returns None."""
        return None
