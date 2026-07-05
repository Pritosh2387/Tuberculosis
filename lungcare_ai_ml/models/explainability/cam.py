"""
Class Activation Mapping (CAM) for LungCare AI.

Implements the original CAM technique (Zhou et al., 2016) which requires
the model to have a **Global Average Pooling → Linear** structure (no
intermediate dense layers).  Supported architectures:

- ResNet50 (GAP → Linear with 2048 features)
- DenseNet121 (GAP → Linear with 1024 features)
- EfficientNet-B0 (GAP → Linear with 1280 features)

For Vision Transformers, use :class:`AttentionRollout` instead.
For arbitrary architectures, use :class:`GradCAM`.

Reference
---------
Zhou et al., "Learning Deep Features for Discriminative Localization",
CVPR 2016.  https://arxiv.org/abs/1512.04150
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("lungcare.explainability.cam")


class CAM:
    """
    Class Activation Maps for GAP + Linear classification heads.

    For each target class *c*, the CAM is computed as:

    .. math::
        M_c(x, y) = \\text{ReLU}\\!\\left(\\sum_k w^c_k \\cdot A_k(x, y)\\right)

    where :math:`A_k` are the spatial feature maps from ``get_features(x)``
    and :math:`w^c_k` are the weights of the final Linear layer for class *c*.

    Args:
        model: A :class:`BaseClassifier` instance with ``get_features()``
            and a ``self.classifier`` attribute whose last module is
            :class:`torch.nn.Linear`.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._fc_weights: torch.Tensor = self._extract_fc_weights()

    def _extract_fc_weights(self) -> torch.Tensor:
        """Extract weight matrix from the final Linear layer of ``self.classifier``."""
        classifier = getattr(self.model, "classifier", None)
        if classifier is None:
            raise AttributeError(
                "Model must have a 'classifier' attribute containing the head."
            )
        linear: nn.Linear | None = None
        if isinstance(classifier, nn.Linear):
            linear = classifier
        elif isinstance(classifier, nn.Sequential):
            for module in reversed(list(classifier.children())):
                if isinstance(module, nn.Linear):
                    linear = module
                    break
        if linear is None:
            raise TypeError(
                "Could not locate an nn.Linear layer inside model.classifier."
            )
        return linear.weight.data  # (num_classes, num_features)

    @torch.no_grad()
    def compute(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Compute the CAM for a single input image.

        Args:
            input_tensor: Preprocessed image tensor ``(1, C, H, W)``.
            target_class: Class index to visualise.  If ``None``, uses
                the predicted class (argmax of logits).
            output_size: ``(height, width)`` to resize the heatmap.
                If ``None``, returns the feature map spatial size.

        Returns:
            ``(cam, target_class)`` where *cam* is a ``float32`` array
            in ``[0, 1]`` of shape ``output_size`` (or feature map size).
        """
        self.model.eval()
        if not hasattr(self.model, "get_features"):
            raise AttributeError("Model must implement get_features().")

        features: torch.Tensor = self.model.get_features(input_tensor)  # (1, C, H, W)
        logits: torch.Tensor = self.model(input_tensor)                  # (1, num_classes)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        weights = self._fc_weights[target_class].to(features.device)    # (C,)

        # Weighted sum over channel dimension
        cam = (weights[:, None, None] * features[0]).sum(dim=0)         # (H, W)
        cam = torch.relu(cam).cpu().numpy().astype(np.float32)

        cam = self._normalize(cam)
        if output_size is not None:
            cam = cv2.resize(cam, (output_size[1], output_size[0]))

        return cam, target_class

    def compute_batch(
        self,
        input_batch: torch.Tensor,
        target_classes: list[int | None] | None = None,
        output_size: tuple[int, int] | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """
        Compute CAMs for a batch of images.

        Args:
            input_batch: Tensor ``(B, C, H, W)``.
            target_classes: List of target class indices (``None`` = predicted).
            output_size: Resize output heatmap.

        Returns:
            List of ``(cam, target_class)`` tuples, one per image.
        """
        B = input_batch.shape[0]
        if target_classes is None:
            target_classes = [None] * B
        return [
            self.compute(input_batch[i : i + 1], target_classes[i], output_size)
            for i in range(B)
        ]

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        """Normalise a CAM array to [0, 1]."""
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            return np.zeros_like(cam)
        return (cam - cam_min) / (cam_max - cam_min)

    def overlay_on_image(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay a CAM heatmap on an RGB image.

        Args:
            image: Original image ``(H, W, 3)`` uint8.
            cam: Normalised CAM ``(H, W)`` float32 in [0, 1].
            alpha: Blending weight for the heatmap overlay.
            colormap: OpenCV colormap constant.

        Returns:
            Blended ``(H, W, 3)`` uint8 image.
        """
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), colormap
        )
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1 - alpha, heatmap_rgb, alpha, 0)
