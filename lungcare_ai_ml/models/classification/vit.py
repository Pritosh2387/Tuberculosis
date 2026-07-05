"""
Vision Transformer (ViT-Base/16) classifier for LungCare AI.

Uses ``timm`` to load a pretrained ViT-Base/16 backbone.  The built-in
classification head is removed (``num_classes=0``) so the model returns
the ``[CLS]`` token embedding ``(B, 768)``.  A custom LayerNorm + Linear
head is appended.

Explainability note
-------------------
ViT does **not** produce spatial conv feature maps, so standard Grad-CAM
is suboptimal.  Use :class:`models.explainability.attention_map.AttentionRollout`
instead.  ``get_target_layer()`` returns the final transformer block for
Grad-CAM compatibility only.
"""

from __future__ import annotations

import logging

import timm
import torch
import torch.nn as nn

from models.classification.base_classifier import BaseClassifier

logger = logging.getLogger("lungcare.models.vit")

_TIMM_MODEL: str = "vit_base_patch16_224"
_EMBED_DIM: int = 768


class ViTClassifier(BaseClassifier):
    """
    Vision Transformer (ViT-Base/16) chest X-ray classifier.

    Classification head::

        LayerNorm(768) → Dropout → Linear(768, num_classes)

    The model requires exactly 224×224 RGB input.

    Args:
        num_classes: Number of output logits.
        task: ``'binary'``, ``'multiclass'``, or ``'multilabel'``.
        pretrained: Load ImageNet-pretrained weights via ``timm``.
        freeze_backbone: Freeze backbone transformer blocks after construction.
        dropout_rate: Dropout probability in the classification head.
        img_size: Input image size (must match the pretrained model).
        patch_size: Patch size (must match the pretrained model).
    """

    def __init__(
        self,
        num_classes: int = 6,
        task: str = "multiclass",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.3,
        img_size: int = 224,
        patch_size: int = 16,
    ) -> None:
        super().__init__(num_classes=num_classes, task=task, dropout_rate=dropout_rate)

        model_name = f"vit_base_patch{patch_size}_{img_size}"
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        embed_dim: int = self.backbone.num_features
        self.img_size = img_size
        self.patch_size = patch_size
        self._embed_dim = embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(embed_dim, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        num_patches = (img_size // patch_size) ** 2
        logger.info(
            "ViTClassifier | classes=%d | task=%s | embed_dim=%d | "
            "patches=%d | pretrained=%s",
            num_classes, task, embed_dim, num_patches, pretrained,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return raw logits of shape ``(B, num_classes)``.

        Internally: backbone returns CLS token ``(B, embed_dim)`` →
        LayerNorm → Dropout → Linear.
        """
        cls_token = self.backbone(x)
        return self.classifier(cls_token)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the full token sequence ``(B, num_patches + 1, embed_dim)``.

        Index ``[:, 0, :]`` is the ``[CLS]`` token.  Indices
        ``[:, 1:, :]`` are patch tokens used for attention rollout.
        """
        return self.backbone.forward_features(x)

    def get_target_layer(self) -> nn.Module:
        """
        Return the final transformer block for hook-based attribution.

        For highest-quality explainability, prefer
        :class:`models.explainability.attention_map.AttentionRollout`.
        """
        return self.backbone.blocks[-1]  # type: ignore[index]

    def get_attention_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract raw attention weight tensors from every transformer block.

        Each tensor has shape ``(B, num_heads, N+1, N+1)`` where
        ``N = (img_size // patch_size)²``.

        Returns:
            List of attention tensors, one per block (12 for ViT-Base).
        """
        attention_maps: list[torch.Tensor] = []

        hooks = []
        for block in self.backbone.blocks:
            def _hook(m: nn.Module, inp: tuple, _out: torch.Tensor) -> None:
                attention_maps.append(inp[0].detach())

            hooks.append(block.attn.attn_drop.register_forward_pre_hook(_hook))

        with torch.no_grad():
            _ = self.backbone.forward_features(x)

        for h in hooks:
            h.remove()

        return attention_maps
