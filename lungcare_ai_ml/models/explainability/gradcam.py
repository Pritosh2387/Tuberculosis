"""
Grad-CAM and Grad-CAM++ for LungCare AI.

Both methods use forward/backward hooks to compute class-discriminative
activation maps without modifying the model architecture.  They work with
any CNN-based :class:`BaseClassifier` as long as a target layer can be
identified via ``model.get_target_layer()``.

Grad-CAM
--------
Selbach et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

Grad-CAM++
----------
Chattopadhyay et al., "Grad-CAM++: Improved Visual Explanations for
Deep Convolutional Networks", WACV 2018.

Usage
-----
With context manager (recommended — auto-removes hooks)::

    with GradCAM(model) as cam:
        heatmap, cls_idx = cam.compute(input_tensor)
        overlay = cam.overlay_on_image(image, heatmap)

Or manually::

    cam = GradCAM(model)
    heatmap, cls_idx = cam.compute(input_tensor)
    cam.remove_hooks()
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("lungcare.explainability.gradcam")


class _HookStore:
    """Stores forward activations and backward gradients from a single layer."""

    def __init__(self) -> None:
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._hooks: list[Any] = []

    def register(self, layer: nn.Module) -> None:
        def _fwd_hook(_m: nn.Module, _inp: Any, output: torch.Tensor) -> None:
            self.activations = output.detach()

        def _bwd_hook(_m: nn.Module, _inp: Any, grad_output: tuple) -> None:
            self.gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(_fwd_hook))
        self._hooks.append(layer.register_full_backward_hook(_bwd_hook))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self) -> None:
        self.activations = None
        self.gradients = None


class GradCAM:
    """
    Grad-CAM: gradient-weighted class activation maps.

    Args:
        model: A :class:`BaseClassifier` instance.
        target_layer: The layer to hook.  Defaults to
            ``model.get_target_layer()`` if ``None``.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
    ) -> None:
        self.model = model
        layer = target_layer or model.get_target_layer()  # type: ignore[attr-defined]
        self._store = _HookStore()
        self._store.register(layer)

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute a Grad-CAM heatmap for one image.

        Args:
            input_tensor: ``(1, C, H, W)`` float tensor (requires_grad not needed).
            target_class: Class to explain.  ``None`` → predicted class.
            output_size: ``(H, W)`` to resize the heatmap.

        Returns:
            ``(heatmap, target_class)`` where *heatmap* is ``float32``
            in ``[0, 1]`` of shape *output_size* (or feature map size).
        """
        self.model.eval()
        self._store.clear()

        inp = input_tensor.clone().requires_grad_(True)
        logits = self.model(inp)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        grads = self._store.gradients      # (1, C, H, W)
        acts = self._store.activations     # (1, C, H, W)

        if grads is None or acts is None:
            raise RuntimeError("Hooks did not capture gradients / activations.")

        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = torch.relu((weights * acts).sum(dim=1)).squeeze()  # (H, W)
        heatmap = self._postprocess(cam, output_size)
        return heatmap, target_class

    @staticmethod
    def _postprocess(
        cam: torch.Tensor,
        output_size: tuple[int, int] | None,
    ) -> np.ndarray:
        h = cam.cpu().numpy().astype(np.float32)
        cam_min, cam_max = h.min(), h.max()
        h = (h - cam_min) / (cam_max - cam_min + 1e-8)
        if output_size is not None:
            h = cv2.resize(h, (output_size[1], output_size[0]))
        return h

    def remove_hooks(self) -> None:
        """Remove all registered forward / backward hooks."""
        self._store.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_: Any) -> None:
        self.remove_hooks()

    @staticmethod
    def overlay_on_image(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blend a normalised heatmap over an RGB image.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image.
            heatmap: ``(H, W)`` float32 heatmap in ``[0, 1]``.
            alpha: Heatmap blending weight.
            colormap: OpenCV colormap constant.

        Returns:
            ``(H, W, 3)`` uint8 blended image.
        """
        h = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heat_uint8 = cv2.applyColorMap((h * 255).astype(np.uint8), colormap)
        heat_rgb = cv2.cvtColor(heat_uint8, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1.0 - alpha, heat_rgb, alpha, 0)


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: improved spatial localisation via second-order gradients.

    Uses the Chattopadhyay et al. alpha weighting scheme which assigns
    higher weights to gradients in regions more strongly associated with
    the target class, yielding sharper and more complete heatmaps.
    """

    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute a Grad-CAM++ heatmap for one image.

        Args:
            input_tensor: ``(1, C, H, W)`` float tensor.
            target_class: Class to explain.  ``None`` → predicted class.
            output_size: ``(H, W)`` to resize the output.

        Returns:
            ``(heatmap, target_class)``.
        """
        self.model.eval()
        self._store.clear()

        inp = input_tensor.clone().requires_grad_(True)
        logits = self.model(inp)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        self.model.zero_grad()
        logits[0, target_class].backward()

        grads = self._store.gradients   # (1, C, H, W)
        acts = self._store.activations  # (1, C, H, W)

        if grads is None or acts is None:
            raise RuntimeError("Hooks did not capture gradients / activations.")

        # Grad-CAM++ alpha weights (Chattopadhyay et al. eq. 19)
        grads_sq = grads ** 2
        grads_cb = grads ** 3
        denom = 2.0 * grads_sq + acts * grads_cb
        denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
        alpha = grads_sq / denom

        # Spatial weights: sum of (alpha * relu(grads)) per channel
        weights = (alpha * torch.relu(grads)).sum(
            dim=(2, 3), keepdim=True
        )   # (1, C, 1, 1)

        cam = torch.relu((weights * acts).sum(dim=1)).squeeze()  # (H, W)
        heatmap = self._postprocess(cam, output_size)
        return heatmap, target_class
