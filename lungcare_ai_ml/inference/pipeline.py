"""
inference/pipeline.py
──────────────────────
End-to-end LungCare AI inference pipeline.

LungCarePipeline is the single public entry point for production inference.

Steps performed by predict()
-----------------------------
1. Load image (JPEG/PNG) and apply preprocessing transforms
2. Classify → softmax probabilities + predicted class + confidence
3. Grad-CAM or AttentionRollout → heatmap (H, W) float32
4. Optional UNet segmentation → binary mask
5. Overlay heatmap on original image (RGB visualisation)
6. Generate structured JSON report via ReportGenerator

Usage
-----
    from inference.pipeline import LungCarePipeline
    from utils.config import load_config

    cfg      = load_config("config.yaml")
    pipeline = LungCarePipeline.from_config(cfg, classifier)
    result   = pipeline.predict("data/patient_001.png")
    print(result.report)
"""
from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from evaluation.report_generator import ReportGenerator

logger = logging.getLogger("lungcare.inference.pipeline")

# ImageNet normalisation constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Postprocessing helpers (inlined from deleted postprocessing.py) ──────────

def decode_classification(
    logits: torch.Tensor,
    class_names: list[str],
    task: str = "multiclass",
    threshold: float = 0.5,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Convert raw logits to a structured classification result dict.

    Args:
        logits:      Raw model output (1, num_classes) or (1,) for binary.
        class_names: Ordered list of class name strings.
        task:        ``'multiclass'`` or ``'binary'``.
        threshold:   Sigmoid threshold for binary classification.
        top_k:       Number of top predictions to include.

    Returns:
        Dict with keys: pred_class, confidence, all_probs, top_k_preds.
    """
    logits = logits.detach().cpu()

    if task == "multiclass":
        probs    = torch.softmax(logits, dim=1).squeeze(0)  # (C,)
        pred_idx = int(probs.argmax().item())
        conf     = float(probs[pred_idx].item())
    else:
        prob     = torch.sigmoid(logits).squeeze().item()
        pred_idx = 1 if prob >= threshold else 0
        probs    = torch.tensor([1 - prob, prob])
        conf     = float(prob if pred_idx == 1 else 1 - prob)

    all_probs = {cls: round(float(p), 4)
                 for cls, p in zip(class_names, probs.tolist())}

    top_k_actual = min(top_k, len(class_names))
    top_k_preds  = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:top_k_actual]

    return {
        "pred_class":  class_names[pred_idx],
        "pred_idx":    pred_idx,
        "confidence":  round(conf, 4),
        "all_probs":   all_probs,
        "top_k_preds": top_k_preds,
    }


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap onto an RGB image.

    Args:
        image_rgb: Original (H, W, 3) uint8 RGB image.
        heatmap:   Float32 (H, W) in [0, 1].
        alpha:     Heatmap opacity.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    heat_u8    = (heatmap * 255).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    if image_rgb.shape[:2] != heat_color.shape[:2]:
        heat_color = cv2.resize(heat_color,
                                (image_rgb.shape[1], image_rgb.shape[0]))

    return cv2.addWeighted(image_rgb, 1 - alpha, heat_color, alpha, 0).astype(np.uint8)


# ─── Pipeline result ─────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Container for all outputs from one LungCarePipeline.predict() call.

    Attributes:
        report:           Structured clinical report dict (JSON-serialisable).
        classification:   Raw decode_classification() output dict.
        heatmap:          Float32 (H, W) Grad-CAM heatmap or None.
        segmentation_mask: Binary uint8 (H, W) mask or None.
        overlay_image:    RGB uint8 (H, W, 3) image with heatmap overlay.
        original_image:   Original RGB uint8 numpy array.
        inference_time_s: Total wall-clock time in seconds.
    """
    report:            dict[str, Any]
    classification:    dict[str, Any]
    heatmap:           np.ndarray | None = None
    segmentation_mask: np.ndarray | None = None
    overlay_image:     np.ndarray | None = None
    original_image:    np.ndarray | None = None
    inference_time_s:  float             = 0.0

    def gradcam_as_b64_png(self) -> str | None:
        """Return the overlay image as a base64-encoded PNG string for API responses."""
        if self.overlay_image is None:
            return None
        img_bgr = cv2.cvtColor(self.overlay_image, cv2.COLOR_RGB2BGR)
        ok, buf  = cv2.imencode(".png", img_bgr)
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class LungCarePipeline:
    """
    End-to-end LungCare AI inference pipeline.

    Args:
        classifier:        Trained classification model (eval mode set internally).
        class_names:       Ordered disease class name list.
        image_size:        Resize target for preprocessing.
        device:            Inference device string.
        explainability:    ``'gradcam'``, ``'rollout'``, or ``'none'``.
        seg_model:         Optional trained UNet segmentation model.
        run_segmentation:  Whether to run segmentation.
        threshold:         Sigmoid threshold for binary/segmentation.
        model_version:     Version string embedded in reports.
    """

    def __init__(
        self,
        classifier: nn.Module,
        class_names: list[str],
        image_size: int = 224,
        device: str = "cpu",
        explainability: str = "gradcam",
        seg_model: nn.Module | None = None,
        run_segmentation: bool = False,
        threshold: float = 0.5,
        model_version: str = "v2.0",
    ) -> None:
        self.device          = torch.device(device)
        self._device_type    = "cuda" if self.device.type == "cuda" else "cpu"
        self.class_names     = class_names
        self.image_size      = image_size
        self.explainability  = explainability
        self.run_segmentation = run_segmentation
        self.threshold       = threshold

        self.classifier = classifier.to(self.device).eval()
        self.seg_model  = seg_model.to(self.device).eval() if seg_model else None

        self.report_gen = ReportGenerator(
            class_names=class_names,
            model_version=model_version,
            finding_threshold=threshold,
        )

        logger.info(
            "LungCarePipeline ready | device=%s | explainability=%s | seg=%s",
            device, explainability, seg_model is not None,
        )

    @classmethod
    def from_config(
        cls,
        config: Any,
        classifier: nn.Module,
        seg_model: nn.Module | None = None,
    ) -> "LungCarePipeline":
        """
        Construct a pipeline from a Config object.

        Args:
            config:     utils.config.Config instance.
            classifier: Trained classifier.
            seg_model:  Optional trained UNet.

        Returns:
            Fully initialised LungCarePipeline.
        """
        ic = config.inference
        return cls(
            classifier=classifier,
            class_names=config.data.class_names,
            image_size=config.data.image_size,
            device=ic.device,
            explainability=ic.explainability,
            seg_model=seg_model,
            run_segmentation=ic.run_segmentation,
            threshold=ic.threshold,
            model_version=config.project.version,
        )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _load_and_preprocess(
        self,
        image_input: str | Path | np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """
        Load an image and return (original_rgb, preprocessed_tensor).

        Args:
            image_input: File path or pre-loaded (H, W, 3) uint8 RGB array.

        Returns:
            Tuple of (original uint8 RGB array, (1, 3, H, W) float32 tensor).
        """
        if isinstance(image_input, (str, Path)):
            pil_img  = Image.open(image_input).convert("RGB")
            original = np.array(pil_img, dtype=np.uint8)
        else:
            original = np.asarray(image_input, dtype=np.uint8)

        # Resize
        resized = cv2.resize(original, (self.image_size, self.image_size))

        # Normalize (ImageNet stats)
        normalized = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD

        # HWC → CHW, add batch dim
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return original, tensor.to(self.device)

    # ── Explainability ────────────────────────────────────────────────────────

    def _compute_heatmap(
        self,
        input_tensor: torch.Tensor,
        pred_idx: int,
        original_size: tuple[int, int],
    ) -> np.ndarray | None:
        from models.gradcam import AttentionRollout, GradCAM  # lazy
        if self.explainability == "none":
            return None

        try:
            if self.explainability == "rollout":
                with AttentionRollout(self.classifier) as rollout:
                    return rollout.compute(input_tensor, output_size=original_size)

            # Default: gradcam
            target_layer = getattr(self.classifier, "get_target_layer", lambda: None)()
            if target_layer is None:
                logger.warning(
                    "Model returned None from get_target_layer(). "
                    "Skipping Grad-CAM (use explainability='rollout' for ViT)."
                )
                return None

            with GradCAM(self.classifier, device=self.device) as cam:
                heatmap, _ = cam.compute(
                    input_tensor,
                    class_idx=pred_idx,
                    output_size=original_size,
                )
            return heatmap

        except Exception as exc:
            logger.warning("Explainability failed: %s", exc)
            return None

    # ── Segmentation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _run_segmentation(
        self,
        input_tensor: torch.Tensor,
        original_size: tuple[int, int],
    ) -> np.ndarray | None:
        if self.seg_model is None:
            return None
        logits = self.seg_model(input_tensor)          # (1, 1, H, W)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
        mask   = (prob >= self.threshold).astype(np.uint8)
        if mask.shape != original_size:
            mask = cv2.resize(mask, (original_size[1], original_size[0]),
                              interpolation=cv2.INTER_NEAREST)
        return mask

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        image_input: str | Path | np.ndarray,
        case_id: str | None = None,
    ) -> PipelineResult:
        """
        Run the full inference pipeline on one image.

        Args:
            image_input: File path (JPEG/PNG) or pre-loaded RGB uint8 array.
            case_id:     Optional patient/case identifier.

        Returns:
            PipelineResult with classification, heatmap, mask, and report.
        """
        t0 = time.time()

        # 1. Load + preprocess
        original_img, input_tensor = self._load_and_preprocess(image_input)
        orig_h, orig_w = original_img.shape[:2]

        # 2. Classification
        with torch.no_grad():
            logits = self.classifier(input_tensor)

        cls_result = decode_classification(
            logits=logits,
            class_names=self.class_names,
            task="multiclass",
            threshold=self.threshold,
        )

        # 3. Explainability — GradCAM needs gradients, so done outside no_grad
        heatmap = self._compute_heatmap(
            input_tensor.detach().requires_grad_(True),
            pred_idx=cls_result["pred_idx"],
            original_size=(orig_h, orig_w),
        )

        # 4. Segmentation (optional)
        seg_mask: np.ndarray | None = None
        if self.run_segmentation and self.seg_model is not None:
            seg_mask = self._run_segmentation(input_tensor, (orig_h, orig_w))

        # 5. Overlay
        overlay: np.ndarray | None = None
        if heatmap is not None:
            overlay = overlay_heatmap(original_img, heatmap)

        # 6. Report
        probs_array = np.array(list(cls_result["all_probs"].values()),
                               dtype=np.float32)
        report = self.report_gen.generate(
            probs=probs_array,
            heatmap=heatmap,
            case_id=case_id,
        )

        return PipelineResult(
            report=report,
            classification=cls_result,
            heatmap=heatmap,
            segmentation_mask=seg_mask,
            overlay_image=overlay,
            original_image=original_img,
            inference_time_s=round(time.time() - t0, 3),
        )

    def save_result(
        self,
        result: PipelineResult,
        output_dir: Path,
        case_id: str = "case",
    ) -> dict[str, Path]:
        """
        Save all pipeline outputs to output_dir.

        Returns:
            Dict of saved file paths keyed by type (report, overlay, mask).
        """
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        # JSON report
        report_path = output_dir / f"{case_id}_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result.report, f, indent=2, default=str)
        saved["report"] = report_path

        # Overlay image
        if result.overlay_image is not None:
            overlay_path = output_dir / f"{case_id}_gradcam.png"
            overlay_bgr  = cv2.cvtColor(result.overlay_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_path), overlay_bgr)
            saved["overlay"] = overlay_path

        # Segmentation mask
        if result.segmentation_mask is not None:
            mask_path = output_dir / f"{case_id}_mask.png"
            cv2.imwrite(str(mask_path),
                        (result.segmentation_mask * 255).astype(np.uint8))
            saved["mask"] = mask_path

        logger.info("Results saved to %s", output_dir)
        return saved
