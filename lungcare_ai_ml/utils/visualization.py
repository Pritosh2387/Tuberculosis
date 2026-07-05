"""
Visualisation utilities for LungCare AI.

Covers heatmap overlays (for Grad-CAM output), confusion matrices, ROC curves,
segmentation comparisons, prediction report cards, and training history plots.

All matplotlib figures use the ``Agg`` (non-interactive) backend so the module
works on headless Linux servers and Windows machines without a display server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("lungcare.visualization")

_COLORMAP_MAP: dict[str, int] = {
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot": cv2.COLORMAP_HOT,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "bone": cv2.COLORMAP_BONE,
}


def _resolve_colormap(colormap: str | int) -> int:
    """Resolve a colormap name string or cv2 constant to a cv2 constant."""
    if isinstance(colormap, int):
        return colormap
    key = colormap.lower()
    if key not in _COLORMAP_MAP:
        logger.warning(
            "Unknown colormap '%s', falling back to 'jet'.", colormap
        )
        return cv2.COLORMAP_JET
    return _COLORMAP_MAP[key]


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert any 2D or 3D float/uint array to uint8 RGB."""
    img = image.copy()
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).clip(0, 255)
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    return img


# ─── Heatmap overlays ─────────────────────────────────────────────────────────


def apply_colormap(
    heatmap: np.ndarray,
    colormap: str | int = "jet",
) -> np.ndarray:
    """
    Apply a colour map to a single-channel heatmap.

    Args:
        heatmap: Float array in ``[0, 1]`` of shape ``(H, W)``.
        colormap: Colormap name (``'jet'``, ``'inferno'``, etc.) or cv2
            constant (``cv2.COLORMAP_JET``).

    Returns:
        ``uint8`` BGR image of shape ``(H, W, 3)``.
    """
    norm = (heatmap * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, _resolve_colormap(colormap))


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str | int = "jet",
) -> np.ndarray:
    """
    Blend a heatmap over an image.

    Args:
        image: Base image (any dtype, greyscale or RGB).
        heatmap: Normalised activation map in ``[0, 1]``, shape ``(H, W)``.
        alpha: Weight of the heatmap in the blend (0 = image only,
            1 = heatmap only).
        colormap: Heatmap colour map.

    Returns:
        ``uint8`` BGR overlay of the same spatial size as *image*.
    """
    base = _ensure_uint8_rgb(image)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)

    h, w = base_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h))
    colored = apply_colormap(heatmap_resized, colormap)

    return cv2.addWeighted(base_bgr, 1 - alpha, colored, alpha, 0)


def save_heatmap(
    heatmap: np.ndarray,
    path: Path | str,
    colormap: str | int = "jet",
) -> None:
    """
    Save a coloured heatmap to disk.

    Args:
        heatmap: Normalised array in ``[0, 1]``, shape ``(H, W)``.
        path: Output file path (extension determines format).
        colormap: Heatmap colour map.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colored = apply_colormap(heatmap, colormap)
    cv2.imwrite(str(out_path), colored)
    logger.debug("Heatmap saved: %s", out_path.name)


def save_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    path: Path | str,
    alpha: float = 0.4,
    colormap: str | int = "jet",
) -> None:
    """
    Save a heatmap-over-image overlay to disk.

    Args:
        image: Base image (any dtype, greyscale or RGB).
        heatmap: Normalised activation map in ``[0, 1]``, shape ``(H, W)``.
        path: Output file path.
        alpha: Heatmap blend weight.
        colormap: Heatmap colour map.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
    cv2.imwrite(str(out_path), overlay)
    logger.debug("Overlay saved: %s", out_path.name)


# ─── Classification visualisations ────────────────────────────────────────────


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: Path | str | None = None,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (10, 8),
    normalize: bool = True,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot a styled confusion matrix with per-cell annotations.

    Args:
        cm: Integer confusion matrix of shape ``(N, N)``.
        class_names: List of N class label strings.
        save_path: If provided, save the figure to this path.
        title: Figure title.
        figsize: Matplotlib figure size ``(width, height)`` in inches.
        normalize: If ``True``, display row-normalised percentages.
        cmap: Seaborn colour palette name.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        display_data = cm.astype(float) / row_sums
        fmt_str = ".2f"
    else:
        display_data = cm.astype(float)
        fmt_str = ".0f"

    sns.heatmap(
        display_data,
        annot=True,
        fmt=fmt_str,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0 if normalize else display_data.max(),
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)

    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved: %s", out.name)

    return fig


def plot_roc_curves(
    fpr_dict: dict[str, np.ndarray],
    tpr_dict: dict[str, np.ndarray],
    auc_dict: dict[str, float],
    save_path: Path | str | None = None,
    title: str = "ROC Curves (One-vs-Rest)",
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot multi-class ROC curves on a single axes.

    Args:
        fpr_dict: Mapping from class name to FPR array.
        tpr_dict: Mapping from class name to TPR array.
        auc_dict: Mapping from class name to AUC scalar.
        save_path: If provided, save the figure here.
        title: Figure title.
        figsize: Figure size.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colours = plt.cm.get_cmap("tab10", len(fpr_dict))

    for idx, class_name in enumerate(fpr_dict):
        ax.plot(
            fpr_dict[class_name],
            tpr_dict[class_name],
            color=colours(idx),
            linewidth=2,
            label=f"{class_name} (AUC = {auc_dict.get(class_name, 0.0):.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("ROC curve plot saved: %s", out.name)

    return fig


def plot_training_history(
    history: dict[str, list[float]],
    save_path: Path | str | None = None,
    title: str = "Training History",
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot training and validation loss / metric curves.

    Args:
        history: Dict mapping metric names to lists of per-epoch values.
            Expected keys follow the pattern ``'train_loss'``, ``'val_loss'``,
            ``'train_f1'``, ``'val_f1'``, etc.
        save_path: If provided, save the figure here.
        title: Overall figure title.
        figsize: Figure size.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    loss_keys = [k for k in history if "loss" in k.lower()]
    metric_keys = [k for k in history if "loss" not in k.lower()]

    n_panels = (1 if loss_keys else 0) + (1 if metric_keys else 0)
    if n_panels == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    panel = 0
    if loss_keys:
        ax = axes[panel]
        for key in loss_keys:
            ax.plot(history[key], label=key.replace("_", " ").title(), linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(alpha=0.3)
        panel += 1

    if metric_keys:
        ax = axes[panel]
        for key in metric_keys:
            ax.plot(history[key], label=key.replace("_", " ").title(), linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Metrics")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("Training history plot saved: %s", out.name)

    return fig


# ─── Segmentation visualisations ──────────────────────────────────────────────


def visualize_segmentation(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray | None = None,
    save_path: Path | str | None = None,
    title: str = "Segmentation Result",
    figsize: tuple[int, int] = (15, 5),
    mask_alpha: float = 0.45,
) -> plt.Figure:
    """
    Side-by-side visualisation: image | (optional) ground truth | prediction.

    Args:
        image: Input image, greyscale or RGB, any dtype.
        pred_mask: Binary or probability mask, shape ``(H, W)``.
        true_mask: Ground truth binary mask.  If ``None`` only two panels
            are shown.
        save_path: If provided, save the figure here.
        title: Figure title.
        figsize: Figure size.
        mask_alpha: Opacity of the mask overlay.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    img_rgb = _ensure_uint8_rgb(image)
    pred_bin = (pred_mask > 0.5).astype(np.uint8)

    n_panels = 3 if true_mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    panel = 1
    if true_mask is not None:
        true_bin = (true_mask > 0.5).astype(np.uint8)
        overlay = img_rgb.copy()
        overlay[true_bin == 1] = (
            overlay[true_bin == 1] * (1 - mask_alpha) + np.array([0, 255, 0]) * mask_alpha
        ).astype(np.uint8)
        axes[panel].imshow(overlay)
        axes[panel].set_title("Ground Truth")
        axes[panel].axis("off")
        panel += 1

    pred_overlay = img_rgb.copy()
    pred_overlay[pred_bin == 1] = (
        pred_overlay[pred_bin == 1] * (1 - mask_alpha)
        + np.array([255, 80, 80]) * mask_alpha
    ).astype(np.uint8)
    axes[panel].imshow(pred_overlay)
    axes[panel].set_title("Prediction")
    axes[panel].axis("off")

    gt_patch = mpatches.Patch(color="lime", label="Ground truth")
    pred_patch = mpatches.Patch(color="salmon", label="Prediction")
    legend_handles = [pred_patch] if true_mask is None else [gt_patch, pred_patch]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(legend_handles),
        fontsize=9, frameon=True
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("Segmentation visualisation saved: %s", out.name)

    return fig


# ─── Prediction report card ───────────────────────────────────────────────────


def create_prediction_card(
    image: np.ndarray,
    findings: dict[str, Any],
    heatmap: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    save_path: Path | str | None = None,
    figsize: tuple[int, int] = (18, 6),
    heatmap_alpha: float = 0.4,
    colormap: str | int = "jet",
) -> plt.Figure:
    """
    Create a full prediction report card with image, heatmap, mask, and text.

    Args:
        image: Input image (any dtype, greyscale or RGB).
        findings: Dict with keys: ``'prediction'``, ``'confidence'``,
            ``'status'``, and optionally ``'findings'`` (list of strings).
        heatmap: Optional normalised activation map ``(H, W)`` in ``[0, 1]``.
        mask: Optional binary segmentation mask ``(H, W)``.
        save_path: If provided, save the figure here.
        figsize: Figure size.
        heatmap_alpha: Heatmap blend weight on the overlay panel.
        colormap: Colormap for heatmap overlay.

    Returns:
        The :class:`matplotlib.figure.Figure` object.
    """
    n_panels = 1 + (1 if heatmap is not None else 0) + (1 if mask is not None else 0) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    img_rgb = _ensure_uint8_rgb(image)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    panel = 1
    if heatmap is not None:
        overlay_bgr = overlay_heatmap(image, heatmap, alpha=heatmap_alpha, colormap=colormap)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        axes[panel].imshow(overlay_rgb)
        axes[panel].set_title("Activation Map", fontsize=11, fontweight="bold")
        axes[panel].axis("off")
        panel += 1

    if mask is not None:
        mask_overlay = img_rgb.copy()
        mask_bin = (mask > 0.5).astype(np.uint8)
        mask_overlay[mask_bin == 1] = (
            mask_overlay[mask_bin == 1] * 0.55 + np.array([255, 80, 80]) * 0.45
        ).astype(np.uint8)
        axes[panel].imshow(mask_overlay)
        axes[panel].set_title("Segmentation", fontsize=11, fontweight="bold")
        axes[panel].axis("off")
        panel += 1

    ax_text = axes[panel]
    ax_text.axis("off")

    prediction = findings.get("prediction", "Unknown")
    confidence = findings.get("confidence", 0.0)
    status = findings.get("status", "unknown").upper()
    finding_list: list[str] = findings.get("findings", [])

    status_colour = "#e74c3c" if status == "ABNORMAL" else "#27ae60"
    y_cursor = 0.95

    ax_text.text(
        0.05, y_cursor, "LungCare AI Report",
        transform=ax_text.transAxes,
        fontsize=13, fontweight="bold", va="top",
    )
    y_cursor -= 0.12

    ax_text.text(
        0.05, y_cursor, f"Status: {status}",
        transform=ax_text.transAxes,
        fontsize=11, color=status_colour, va="top", fontweight="bold",
    )
    y_cursor -= 0.10

    ax_text.text(
        0.05, y_cursor, f"Diagnosis: {prediction}",
        transform=ax_text.transAxes, fontsize=10, va="top",
    )
    y_cursor -= 0.09

    ax_text.text(
        0.05, y_cursor, f"Confidence: {confidence * 100:.1f}%",
        transform=ax_text.transAxes, fontsize=10, va="top",
        color="#2c3e50",
    )
    y_cursor -= 0.12

    if finding_list:
        ax_text.text(
            0.05, y_cursor, "Findings:",
            transform=ax_text.transAxes, fontsize=10, va="top", fontweight="bold",
        )
        y_cursor -= 0.10
        for finding_str in finding_list:
            ax_text.text(
                0.07, y_cursor, f"• {finding_str}",
                transform=ax_text.transAxes, fontsize=9, va="top",
                wrap=True, color="#34495e",
            )
            y_cursor -= 0.09
            if y_cursor < 0.05:
                break

    ax_text.axhline(
        y=0.88, xmin=0.02, xmax=0.98,
        transform=ax_text.transAxes, color="#bdc3c7", linewidth=0.8,
    )

    fig.suptitle("LungCare AI — Diagnostic Report", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        logger.info("Prediction card saved: %s", out.name)

    return fig
