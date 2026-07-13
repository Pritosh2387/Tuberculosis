"""
models/gradcam.py
──────────────────
Grad-CAM and Attention Rollout for LungCare AI explainability.

GradCAM
-------
Selbach et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

Works on any CNN that exposes get_target_layer(). Uses forward/backward
hooks to capture activations and gradients at the target layer, then
computes a class-discriminative heatmap without modifying the model.

Tensor flow:
    input (1,3,H,W)
    → model.forward()           [captures activations A (1,C,h,w)]
    → loss = logits[class_idx]
    → loss.backward()           [captures gradients G (1,C,h,w)]
    → weights = mean(G, dim=(2,3))          → (C,)
    → cam = relu(sum(weights * A, dim=1))   → (h,w)
    → resize to (H,W) → normalize [0,1]

AttentionRollout
----------------
Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.

Works on ViT (where GradCAM doesn't apply). Propagates attention
weights layer-by-layer to attribute each image patch's contribution
to the [CLS] token classification decision.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("lungcare.models.gradcam")


# ─── Hook storage ─────────────────────────────────────────────────────────────


class _HookStore:
    """Accumulates forward activations and backward gradients for one layer."""

    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

    def save_activation(self) -> Any:
        def hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
            self.activations = out.detach()
        return hook

    def save_gradient(self) -> Any:
        def hook(module: nn.Module, grad_in: Any, grad_out: tuple[torch.Tensor, ...]) -> None:
            self.gradients = grad_out[0].detach()
        return hook


# ─── GradCAM ──────────────────────────────────────────────────────────────────


class GradCAM:
    """
    Grad-CAM for any CNN classifier with a ``get_target_layer()`` method.

    Usage (context manager — automatically removes hooks)::

        with GradCAM(model) as cam:
            heatmap, class_idx = cam.compute(input_tensor)
            overlay = cam.overlay(original_image_rgb, heatmap)

    Args:
        model:      Trained CNN classifier.
        device:     Inference device.

    Interview note: Why context manager?  Pytorch hooks are persistent —
    if not removed they accumulate memory and distort future forward passes.
    The context manager guarantees cleanup even if an exception occurs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self._store = _HookStore()
        self._hooks: list[Any] = []

        target_layer = model.get_target_layer()  # type: ignore[attr-defined]
        if target_layer is None:
            raise ValueError(
                "Model returned None from get_target_layer(). "
                "Use AttentionRollout for ViT models."
            )
        self._register_hooks(target_layer)

    def _register_hooks(self, layer: nn.Module) -> None:
        self._hooks.append(
            layer.register_forward_hook(self._store.save_activation())
        )
        self._hooks.append(
            layer.register_full_backward_hook(self._store.save_gradient())
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    def compute(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute the Grad-CAM heatmap for one image.

        Args:
            input_tensor: Preprocessed image (1, C, H, W).
            class_idx:    Target class index. If None, uses argmax(logits).
            output_size:  (H, W) to resize the heatmap to. Defaults to input size.

        Returns:
            Tuple of (heatmap, class_idx) where heatmap is float32 (H, W) in [0, 1].
        """
        self.model.zero_grad()
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)   # (1, num_classes)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass — gradients w.r.t. target class score
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM weights: global average pool of gradients
        grads = self._store.gradients      # (1, C, h, w)
        acts  = self._store.activations    # (1, C, h, w)

        if grads is None or acts is None:
            raise RuntimeError("Hooks did not capture gradients/activations.")

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam).squeeze().cpu().numpy()     # (h, w)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        h, w = output_size if output_size else (input_tensor.shape[2], input_tensor.shape[3])
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam_resized.astype(np.float32), class_idx

    @staticmethod
    def overlay(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend a Grad-CAM heatmap onto an RGB image.

        Args:
            image_rgb: Original (H, W, 3) uint8 RGB image.
            heatmap:   Float32 (H, W) array in [0, 1].
            alpha:     Heatmap opacity (0 = invisible, 1 = fully opaque).
            colormap:  OpenCV colormap constant.

        Returns:
            (H, W, 3) uint8 blended image.
        """
        heat_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, colormap)
        heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

        if image_rgb.shape[:2] != heat_color.shape[:2]:
            heat_color = cv2.resize(heat_color, (image_rgb.shape[1], image_rgb.shape[0]))

        blended = cv2.addWeighted(image_rgb, 1 - alpha, heat_color, alpha, 0)
        return blended.astype(np.uint8)


# ─── Attention Rollout ────────────────────────────────────────────────────────


class AttentionRollout:
    """
    Attention Rollout for Vision Transformer explainability.

    Propagates raw attention weights through all transformer encoder
    layers to attribute each image patch's contribution to the [CLS]
    token classification decision.

    Requires the timm ViT model to expose ``blocks`` (list of
    transformer blocks), each with ``attn.attn_drop`` as the hook point.

    Args:
        model:           Trained ViTClassifier.
        discard_ratio:   Fraction of lowest-attention tokens to zero out
                         at each layer (reduces noise).
        head_fusion:     How to combine multi-head attention.
                         ``'mean'``, ``'min'``, or ``'max'``.
    """

    def __init__(
        self,
        model: nn.Module,
        discard_ratio: float = 0.9,
        head_fusion: str = "mean",
    ) -> None:
        self.model = model.eval()
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self._attentions: list[torch.Tensor] = []
        self._hooks: list[Any] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        backbone = getattr(self.model, "backbone", self.model)
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            raise ValueError("Model does not expose 'backbone.blocks'. "
                             "Is this a timm ViT model?")
        for block in blocks:
            self._hooks.append(
                block.attn.register_forward_hook(self._store_attention())
            )

    def _store_attention(self) -> Any:
        def hook(module: nn.Module, inp: Any, out: Any) -> None:
            # timm attention returns (output, attn_weights) when
            # attn_drop is called; capture the last call
            if isinstance(out, tuple) and len(out) == 2:
                self._attentions.append(out[1].detach())
        return hook

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self) -> "AttentionRollout":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove_hooks()

    @torch.no_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Compute the Attention Rollout map.

        Args:
            input_tensor: Preprocessed image (1, 3, H, W).
            output_size:  (H, W) to resize the map to.

        Returns:
            Float32 (H, W) attention map in [0, 1].
        """
        self._attentions.clear()
        _ = self.model(input_tensor.to(next(self.model.parameters()).device))

        if not self._attentions:
            raise RuntimeError(
                "No attention weights captured. "
                "Ensure the model uses timm's ViT with attn.attn_drop."
            )

        # Rollout: multiply attention matrices across layers
        result = torch.eye(self._attentions[0].shape[-1])

        for attn in self._attentions:
            # attn: (1, num_heads, num_tokens, num_tokens)
            if self.head_fusion == "mean":
                fused = attn.mean(dim=1).squeeze(0)
            elif self.head_fusion == "max":
                fused = attn.max(dim=1).values.squeeze(0)
            else:  # min
                fused = attn.min(dim=1).values.squeeze(0)

            # Discard lowest-attention tokens (noise reduction)
            flat = fused.flatten()
            threshold = flat.kthvalue(
                max(1, int(self.discard_ratio * flat.size(0)))
            ).values
            fused[fused < threshold] = 0

            # Add residual + normalize
            fused = fused + torch.eye(fused.shape[0])
            fused = fused / fused.sum(dim=-1, keepdim=True)
            result = torch.matmul(fused, result)

        # CLS token row → patch attributions (skip the CLS token itself)
        mask = result[0, 1:]   # (num_patches,) = 196 for ViT-B/16 224x224
        grid_size = int(math.sqrt(mask.size(0)))
        mask = mask.reshape(grid_size, grid_size).numpy()

        if mask.max() > 0:
            mask = mask / mask.max()

        h, w = output_size if output_size else (input_tensor.shape[2], input_tensor.shape[3])
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        return mask_resized.astype(np.float32)
