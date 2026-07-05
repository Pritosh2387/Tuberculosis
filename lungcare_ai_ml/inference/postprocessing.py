"""
Post-processing utilities for LungCare AI inference outputs.

Transforms raw model logits / masks into clinically interpretable outputs:

- Softmax / sigmoid decoding with configurable thresholds.
- Top-K class selection with confidence bounds.
- Binary mask cleaning (morphological open/close, connected-component filtering).
- Contour extraction for bounding-box localisation.
- Heatmap overlaying on original images (supports both CAM and Attention).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger("lungcare.inference.postprocessing")


# ─── Classification ───────────────────────────────────────────────────────────


def decode_classification(
    logits: torch.Tensor,
    class_names: list[str],
    task: str = "multiclass",
    threshold: float = 0.5,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Decode classification logits into a structured prediction dict.

    Args:
        logits: Raw model output ``(1, C)`` or ``(C,)`` tensor.
        class_names: Ordered list of class name strings.
        task: ``'binary'``, ``'multiclass'``, or ``'multilabel'``.
        threshold: Sigmoid decision threshold (binary / multilabel).
        top_k: Number of top classes to include in ``'top_k'`` field.

    Returns:
        Dict with keys:

        - ``'pred_class'``: str (or list for multilabel).
        - ``'confidence'``: float.
        - ``'all_probs'``: {cls: prob}.
        - ``'top_k'``: list of ``{'class': str, 'prob': float}``.
        - ``'is_abnormal'``: bool.
    """
    logits = logits.squeeze(0).float()

    if task == "multiclass":
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_class: str | list[str] = class_names[pred_idx]
        confidence = float(probs[pred_idx])
        is_abnormal = pred_class != "Healthy"

    elif task == "binary":
        prob_pos = float(torch.sigmoid(logits).item())
        probs = np.array([1.0 - prob_pos, prob_pos])
        pred_idx = int(prob_pos >= threshold)
        pred_class = class_names[pred_idx]
        confidence = prob_pos if pred_idx == 1 else 1.0 - prob_pos
        is_abnormal = pred_idx == 1

    else:  # multilabel
        probs = torch.sigmoid(logits).cpu().numpy()
        active = [class_names[i] for i, p in enumerate(probs) if p >= threshold]
        pred_class = active if active else ["Healthy"]
        confidence = float(probs.max())
        is_abnormal = bool(active and active != ["Healthy"])

    # All class probabilities
    all_probs = {
        name: round(float(p), 4)
        for name, p in zip(class_names, probs)
    }

    # Top-K sorted by probability
    sorted_classes = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    top_k_list = [
        {"class": cls, "prob": prob} for cls, prob in sorted_classes[:top_k]
    ]

    return {
        "pred_class": pred_class,
        "confidence": round(confidence, 4),
        "all_probs": all_probs,
        "top_k": top_k_list,
        "is_abnormal": is_abnormal,
    }


# ─── Segmentation ─────────────────────────────────────────────────────────────


def decode_segmentation(
    logits: torch.Tensor,
    original_size: tuple[int, int],
    threshold: float = 0.5,
    min_area_px: int = 100,
    morph_kernel_size: int = 5,
) -> dict[str, Any]:
    """
    Decode segmentation logits into a cleaned binary mask + bounding boxes.

    Pipeline:
    1. Apply sigmoid.
    2. Threshold to binary.
    3. Resize to ``original_size``.
    4. Morphological opening to remove salt noise.
    5. Morphological closing to fill small holes.
    6. Connected-component filtering (remove blobs smaller than *min_area_px*).
    7. Extract bounding boxes and contour areas.

    Args:
        logits: Raw model output ``(1, 1, H, W)`` or ``(1, H, W)``.
        original_size: Target ``(height, width)`` in pixels.
        threshold: Binary threshold on sigmoid probabilities.
        min_area_px: Minimum connected-component area to keep.
        morph_kernel_size: Kernel size for morphological operations.

    Returns:
        Dict with:

        - ``'mask'``: ``(H, W)`` uint8 binary numpy array (0 / 255).
        - ``'bboxes'``: list of ``[x, y, w, h]`` bounding boxes (pixel coords).
        - ``'num_regions'``: number of retained connected components.
        - ``'coverage_pct'``: fraction of image area covered by mask.
        - ``'prob_map'``: ``(H, W)`` float32 probability map in [0, 1].
    """
    logits = logits.squeeze().float()
    prob_map = torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    # Resize probability map to original image size
    H, W = original_size
    prob_resized = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # Binary threshold
    binary = (prob_resized >= threshold).astype(np.uint8) * 255

    # Morphological clean-up
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected-component filtering
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    cleaned = np.zeros_like(binary)
    bboxes: list[list[int]] = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            cleaned[labels == i] = 255
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            bboxes.append([x, y, w, h])

    coverage = float(cleaned.sum()) / (255.0 * H * W)

    return {
        "mask": cleaned,
        "bboxes": bboxes,
        "num_regions": len(bboxes),
        "coverage_pct": round(coverage * 100.0, 2),
        "prob_map": prob_resized,
    }


# ─── Heatmap overlay ──────────────────────────────────────────────────────────


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
    bboxes: list[list[int]] | None = None,
    bbox_color: tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 2,
) -> np.ndarray:
    """
    Blend a normalised heatmap over an image, optionally drawing bounding boxes.

    Args:
        image: RGB uint8 image ``(H, W, 3)``.
        heatmap: Normalised float32 heatmap ``(H, W)`` in [0, 1].
        alpha: Heatmap blend weight.
        colormap: OpenCV colormap constant.
        bboxes: Optional list of ``[x, y, w, h]`` boxes to overlay.
        bbox_color: BGR colour for bounding boxes.
        bbox_thickness: Box line thickness in pixels.

    Returns:
        Blended ``(H, W, 3)`` uint8 image.
    """
    h_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heat_colored = cv2.applyColorMap((h_resized * 255).astype(np.uint8), colormap)
    heat_rgb = cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(image, 1.0 - alpha, heat_rgb, alpha, 0)

    if bboxes:
        for x, y, w, h in bboxes:
            cv2.rectangle(blended, (x, y), (x + w, y + h), bbox_color, bbox_thickness)

    return blended


# ─── Mask overlay ─────────────────────────────────────────────────────────────


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a binary segmentation mask over an RGB image.

    Args:
        image: RGB uint8 image ``(H, W, 3)``.
        mask: Binary uint8 mask ``(H, W)`` with values 0 / 255.
        color: RGB fill colour for the masked region.
        alpha: Mask blend weight.

    Returns:
        Blended ``(H, W, 3)`` uint8 image.
    """
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    mask_bool = mask > 0
    colored_mask[mask_bool] = color
    return cv2.addWeighted(overlay, 1.0 - alpha, colored_mask, alpha, 0)
