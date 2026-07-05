"""
Attention Rollout for Vision Transformer explainability in LungCare AI.

Implements two complementary methods:

1. **AttentionRollout** (Abnar & Zuidema, 2020): Propagates raw attention
   weights through all transformer layers to attribute each input patch's
   contribution to the [CLS] token decision.

2. **AttentionHeatmap**: Lighter alternative — returns only the final
   transformer block's mean attention over heads without rollout.

Reference
---------
Abnar & Zuidema, "Quantifying Attention Flow in Transformers",
ACL 2020.  https://arxiv.org/abs/2005.00928
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("lungcare.explainability.attention_map")


class AttentionRollout:
    """
    Attention Rollout for ViT-based classifiers.

    Hooks into the ``attn_drop`` dropout layer of each transformer block
    to capture the attention weight matrices ``(B, num_heads, N+1, N+1)``.
    Then performs rollout by matrix-multiplying attention maps across layers,
    accounting for residual connections.

    Args:
        model: A :class:`ViTClassifier` instance (must expose
            ``model.backbone.blocks``).
        head_fusion: How to fuse multi-head attention.  One of
            ``'mean'``, ``'max'``, ``'min'``.
        discard_ratio: Fraction of **lowest** attention values set to zero
            before rollout (noise suppression).

    Example::

        with AttentionRollout(model) as rollout:
            heatmap = rollout.compute(input_tensor, grid_size=14)
            overlay = rollout.overlay_on_image(image, heatmap)
    """

    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
    ) -> None:
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._attention_maps: list[torch.Tensor] = []
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register pre-forward hooks on each block's attn_drop layer."""
        blocks = self._get_blocks()
        for block in blocks:
            def _make_hook() -> Any:
                def _hook(m: nn.Module, inp: tuple) -> None:
                    self._attention_maps.append(inp[0].detach().cpu())
                return _hook

            h = block.attn.attn_drop.register_forward_pre_hook(_make_hook())
            self._hooks.append(h)

    def _get_blocks(self) -> list[nn.Module]:
        """Retrieve transformer blocks from the model."""
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "blocks"):
            return list(self.model.backbone.blocks)
        if hasattr(self.model, "blocks"):
            return list(self.model.blocks)
        raise AttributeError(
            "Cannot locate transformer blocks.  "
            "Model must expose 'backbone.blocks' or 'blocks'."
        )

    def _fuse_heads(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Reduce multi-head attention ``(B, H, N, N)`` → ``(B, N, N)``.

        Args:
            attn: Attention tensor with shape ``(B, num_heads, N, N)``.

        Returns:
            Head-fused attention ``(B, N, N)``.
        """
        if self.head_fusion == "mean":
            return attn.mean(dim=1)
        elif self.head_fusion == "max":
            return attn.max(dim=1).values
        elif self.head_fusion == "min":
            return attn.min(dim=1).values
        raise ValueError(f"Unknown head_fusion: '{self.head_fusion}'")

    def _discard_low_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Zero out the lowest ``discard_ratio`` fraction of attention values.

        Args:
            attn: Fused attention ``(B, N, N)``.

        Returns:
            Sparsified attention of the same shape.
        """
        flat = attn.reshape(attn.shape[0], -1)
        k = int(flat.shape[1] * self.discard_ratio)
        threshold, _ = flat.kthvalue(k, dim=1, keepdim=True)
        mask = flat >= threshold.expand_as(flat)
        flat = flat * mask.float()
        return flat.reshape_as(attn)

    @torch.no_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        grid_size: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Compute an attention rollout heatmap.

        Args:
            input_tensor: Preprocessed image tensor ``(1, 3, H, W)``.
            grid_size: Number of patches per spatial dimension
                (``img_size // patch_size``).  Auto-inferred when ``None``.
            output_size: ``(H, W)`` to resize the output heatmap.

        Returns:
            ``float32`` heatmap in ``[0, 1]`` of shape *output_size*
            (or ``(grid_size, grid_size)`` if *output_size* is ``None``).
        """
        self._attention_maps.clear()
        self.model.eval()

        _ = self.model(input_tensor)

        if not self._attention_maps:
            raise RuntimeError(
                "No attention maps captured.  Ensure model has attn_drop layers."
            )

        # Stack: list of (1, num_heads, N, N) → (L, 1, num_heads, N, N)
        result: torch.Tensor | None = None

        for layer_attn in self._attention_maps:
            fused = self._fuse_heads(layer_attn)     # (1, N, N)
            fused = self._discard_low_attention(fused)

            # Add residual skip connection
            N = fused.shape[1]
            identity = torch.eye(N, dtype=fused.dtype).unsqueeze(0)
            attn_with_res = (fused + identity) / 2.0

            # Normalise rows (so attention sums to 1 per token)
            attn_norm = attn_with_res / attn_with_res.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            if result is None:
                result = attn_norm
            else:
                result = torch.bmm(attn_norm, result)

        assert result is not None
        # Extract CLS → patch attention (row 0, columns 1..)
        cls_attn = result[0, 0, 1:]                 # (num_patches,)
        cls_attn = cls_attn.numpy().astype(np.float32)

        if grid_size is None:
            grid_size = int(math.sqrt(cls_attn.shape[0]))

        heatmap = cls_attn.reshape(grid_size, grid_size)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        if output_size is not None:
            heatmap = cv2.resize(heatmap, (output_size[1], output_size[0]))

        return heatmap

    def remove_hooks(self) -> None:
        """Remove all registered attention hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attention_maps.clear()

    def __enter__(self) -> "AttentionRollout":
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove_hooks()

    @staticmethod
    def overlay_on_image(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_INFERNO,
    ) -> np.ndarray:
        """
        Blend an attention heatmap over an RGB image.

        Args:
            image: Original ``(H, W, 3)`` uint8 RGB image.
            heatmap: Normalised ``(H, W)`` float32 attention map.
            alpha: Heatmap blend weight.
            colormap: OpenCV colormap (default ``INFERNO`` suits attention maps).

        Returns:
            Blended ``(H, W, 3)`` uint8 image.
        """
        h = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heat_uint8 = cv2.applyColorMap((h * 255).astype(np.uint8), colormap)
        heat_rgb = cv2.cvtColor(heat_uint8, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1.0 - alpha, heat_rgb, alpha, 0)


class AttentionHeatmap:
    """
    Single-layer attention heatmap (no rollout).

    Faster than :class:`AttentionRollout` — uses only the last transformer
    block's attention.  Suitable for real-time inference.

    Args:
        model: ViT model exposing ``get_attention_weights()``.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @torch.no_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        grid_size: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Compute heatmap from the last layer's mean CLS attention.

        Args:
            input_tensor: ``(1, 3, H, W)`` tensor.
            grid_size: Patches per spatial dimension.
            output_size: Resize output.

        Returns:
            ``float32`` heatmap in ``[0, 1]``.
        """
        self.model.eval()
        if not hasattr(self.model, "get_attention_weights"):
            raise AttributeError("Model must implement get_attention_weights().")

        attn_maps: list[torch.Tensor] = self.model.get_attention_weights(input_tensor)
        last_attn = attn_maps[-1]                   # (1, num_heads, N+1, N+1)
        cls_attn = last_attn[0, :, 0, 1:].mean(0)  # avg over heads → (num_patches,)
        cls_attn = cls_attn.cpu().numpy().astype(np.float32)

        if grid_size is None:
            grid_size = int(math.sqrt(cls_attn.shape[0]))

        heatmap = cls_attn.reshape(grid_size, grid_size)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        if output_size is not None:
            heatmap = cv2.resize(heatmap, (output_size[1], output_size[0]))

        return heatmap
