"""
End-to-end LungCare AI inference pipeline.

:class:`LungCarePipeline` is the single public entry point for
production inference.  It orchestrates:

1. **Pre-processing** — load image (JPEG/PNG/DICOM), apply transforms.
2. **Classification** — multi-disease probability prediction.
3. **Explainability** — Grad-CAM / CAM / Attention Rollout heatmap.
4. **Segmentation** — optional lesion mask prediction.
5. **Healthy comparison** — deviation from healthy reference database.
6. **Report generation** — structured JSON + Markdown clinical report.

All steps are configurable at construction time; unused modules
(segmentation, healthy reference) are optional and may be ``None``.

Usage
-----
.. code-block:: python

    from inference.pipeline import LungCarePipeline, PipelineConfig

    cfg = PipelineConfig.from_yaml("configs/inference_config.yaml")
    pipeline = LungCarePipeline.from_config(cfg)

    result = pipeline.predict("data/patient_001.dcm")
    pipeline.save_result(result, output_dir=Path("results/patient_001"))
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from evaluation.report_generator import ReportGenerator
from inference.postprocessing import (
    decode_classification,
    decode_segmentation,
    overlay_heatmap,
    overlay_mask,
)
from services.healthy_reference import HealthyReferenceDatabase, extract_features

logger = logging.getLogger("lungcare.inference.pipeline")


# ─── Config ───────────────────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """
    Configuration for :class:`LungCarePipeline`.

    All fields can be set from a YAML inference config via
    :meth:`from_yaml`.

    Attributes:
        class_names: Ordered disease class name list.
        task: Classification task type.
        image_size: Resize target ``(H, W)`` for the classifier.
        mean: Normalisation mean.
        std: Normalisation std.
        threshold: Sigmoid threshold (binary / multilabel / segmentation).
        explainability_method: ``'gradcam'``, ``'gradcam++'``, ``'cam'``,
            ``'rollout'``, or ``'none'``.
        run_segmentation: Whether to run the segmentation model.
        amp: Use AMP during inference.
        device: Target device string.
        model_version: Version string for report metadata.
        min_mask_area_px: Minimum connected component area.
    """

    class_names: list[str] = field(
        default_factory=lambda: [
            "Healthy",
            "Tuberculosis",
            "Pneumonia",
            "COVID-19",
            "Lung Cancer",
            "Pulmonary Fibrosis",
        ]
    )
    task: str = "multiclass"
    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    threshold: float = 0.5
    explainability_method: str = "gradcam"
    run_segmentation: bool = False
    amp: bool = False
    device: str = "cpu"
    model_version: str = "v1.0"
    min_mask_area_px: int = 100

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file."""
        import yaml

        with open(yaml_path) as f:
            raw: dict = yaml.safe_load(f)

        cfg = raw.get("inference", raw)
        return cls(
            class_names=cfg.get("class_names", cls.__dataclass_fields__["class_names"].default_factory()),  # type: ignore[misc]
            task=cfg.get("task", "multiclass"),
            image_size=tuple(cfg.get("image_size", [224, 224])),
            mean=tuple(cfg.get("mean", [0.485, 0.456, 0.406])),
            std=tuple(cfg.get("std", [0.229, 0.224, 0.225])),
            threshold=cfg.get("threshold", 0.5),
            explainability_method=cfg.get("explainability_method", "gradcam"),
            run_segmentation=cfg.get("run_segmentation", False),
            amp=cfg.get("amp", False),
            device=cfg.get("device", "cpu"),
            model_version=cfg.get("model_version", "v1.0"),
            min_mask_area_px=cfg.get("min_mask_area_px", 100),
        )


# ─── Pipeline result ─────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """
    Container for all outputs from a single :class:`LungCarePipeline` call.

    Attributes:
        report: Structured clinical report dict.
        classification: Raw decode_classification() output dict.
        heatmap: Normalised float32 heatmap ``(H, W)`` or ``None``.
        segmentation: Raw decode_segmentation() output dict or ``None``.
        overlay_image: RGB uint8 image with heatmap/mask overlay.
        original_image: Original RGB image as numpy array.
        inference_time_s: Total wall-clock time.
    """

    report: dict[str, Any]
    classification: dict[str, Any]
    heatmap: np.ndarray | None = None
    segmentation: dict[str, Any] | None = None
    overlay_image: np.ndarray | None = None
    original_image: np.ndarray | None = None
    inference_time_s: float = 0.0


# ─── Pipeline ─────────────────────────────────────────────────────────────────


class LungCarePipeline:
    """
    End-to-end LungCare AI inference pipeline.

    Args:
        classifier: Trained classification model.
        config: :class:`PipelineConfig` instance.
        segmentation_model: Optional trained segmentation model.
        healthy_db: Optional :class:`HealthyReferenceDatabase`.
    """

    def __init__(
        self,
        classifier: nn.Module,
        config: PipelineConfig,
        segmentation_model: nn.Module | None = None,
        healthy_db: HealthyReferenceDatabase | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._device_type = "cuda" if self.device.type == "cuda" else "cpu"
        self._amp = config.amp and self._device_type == "cuda"

        # Move models to device
        self.classifier = classifier.to(self.device).eval()
        self.seg_model = (
            segmentation_model.to(self.device).eval()
            if segmentation_model is not None
            else None
        )
        self.healthy_db = healthy_db
        self.report_gen = ReportGenerator(
            class_names=config.class_names,
            model_version=config.model_version,
            finding_threshold=config.threshold,
        )

        # Build explainability module
        self._cam_module = self._build_cam(config.explainability_method)

        logger.info(
            "LungCarePipeline ready | device=%s | explainability=%s | seg=%s",
            config.device,
            config.explainability_method,
            segmentation_model is not None,
        )

    # ─── Public interface ─────────────────────────────────────────────────────

    def predict(
        self,
        image_input: str | Path | np.ndarray,
        case_id: str | None = None,
    ) -> PipelineResult:
        """
        Run the full inference pipeline on one scan.

        Args:
            image_input: File path (JPEG/PNG/DICOM) or pre-loaded RGB uint8
                numpy array ``(H, W, 3)``.
            case_id: Optional patient / case identifier.

        Returns:
            :class:`PipelineResult` with all intermediate and final outputs.
        """
        t0 = time.time()

        # ── 1. Load & pre-process ─────────────────────────────────────────────
        original_img, input_tensor = self._load_and_preprocess(image_input)
        original_size = (original_img.shape[0], original_img.shape[1])

        # ── 2. Classification ─────────────────────────────────────────────────
        with torch.no_grad():
            ctx = torch.amp.autocast(device_type=self._device_type, enabled=self._amp)
            with ctx:
                logits: torch.Tensor = self.classifier(input_tensor)

        cls_result = decode_classification(
            logits=logits,
            class_names=self.config.class_names,
            task=self.config.task,
            threshold=self.config.threshold,
        )

        # ── 3. Explainability ─────────────────────────────────────────────────
        heatmap: np.ndarray | None = None
        if self._cam_module is not None:
            try:
                heatmap, _ = self._cam_module.compute(
                    input_tensor, output_size=original_size
                )
            except Exception as exc:
                logger.warning("Explainability failed: %s", exc)

        # ── 4. Segmentation ───────────────────────────────────────────────────
        seg_result: dict[str, Any] | None = None
        if self.seg_model is not None and self.config.run_segmentation:
            seg_result = self._run_segmentation(input_tensor, original_size)

        # ── 5. Healthy comparison ─────────────────────────────────────────────
        patient_feats: np.ndarray | None = None
        healthy_feats: np.ndarray | None = None
        if self.healthy_db is not None:
            try:
                patient_feats = extract_features(
                    self.classifier, input_tensor, device=self.device
                )
                healthy_feats = self.healthy_db.get_mean()
            except Exception as exc:
                logger.warning("Feature extraction failed: %s", exc)

        # ── 6. Report ─────────────────────────────────────────────────────────
        probs_np = np.array(list(cls_result["all_probs"].values()), dtype=np.float32)
        report = self.report_gen.generate(
            probs=probs_np,
            heatmap=heatmap,
            mask_path=None,
            healthy_features=healthy_feats,
            patient_features=patient_feats,
            case_id=case_id,
        )

        # ── 7. Overlay ────────────────────────────────────────────────────────
        overlay: np.ndarray | None = None
        if heatmap is not None:
            overlay = overlay_heatmap(
                original_img.copy(),
                heatmap,
                bboxes=seg_result["bboxes"] if seg_result else None,
            )
        if seg_result is not None and overlay is not None:
            overlay = overlay_mask(overlay, seg_result["mask"], alpha=0.25)
        elif seg_result is not None:
            overlay = overlay_mask(
                original_img.copy(), seg_result["mask"], alpha=0.35
            )

        return PipelineResult(
            report=report,
            classification=cls_result,
            heatmap=heatmap,
            segmentation=seg_result,
            overlay_image=overlay,
            original_image=original_img,
            inference_time_s=round(time.time() - t0, 3),
        )

    # ─── Save helpers ─────────────────────────────────────────────────────────

    def save_result(
        self,
        result: PipelineResult,
        output_dir: Path,
        case_id: str = "case",
    ) -> dict[str, Path]:
        """
        Save all artefacts (report, overlay, mask) to *output_dir*.

        Args:
            result: :class:`PipelineResult` from :meth:`predict`.
            output_dir: Directory to write outputs.
            case_id: Used as the filename base.

        Returns:
            Dict mapping artefact type → saved :class:`Path`.
        """
        import cv2 as _cv2
        import json

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: dict[str, Path] = {}

        # JSON report
        report_path = out / f"{case_id}_report.json"
        with open(report_path, "w") as f:
            json.dump(result.report, f, indent=2, default=str)
        saved["report_json"] = report_path

        # Markdown report
        md_path = out / f"{case_id}_report.md"
        md_path.write_text(self.report_gen.to_markdown(result.report), encoding="utf-8")
        saved["report_md"] = md_path

        # Overlay image
        if result.overlay_image is not None:
            overlay_path = out / f"{case_id}_overlay.png"
            _cv2.imwrite(
                str(overlay_path),
                _cv2.cvtColor(result.overlay_image, _cv2.COLOR_RGB2BGR),
            )
            saved["overlay"] = overlay_path

        # Segmentation mask
        if result.segmentation is not None:
            mask_path = out / f"{case_id}_mask.png"
            _cv2.imwrite(str(mask_path), result.segmentation["mask"])
            saved["mask"] = mask_path
            # Update report with mask path
            result.report["segmentation"]["mask_path"] = str(mask_path)

        logger.info("Results saved → %s (%d files)", out, len(saved))
        return saved

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _load_and_preprocess(
        self,
        image_input: str | Path | np.ndarray,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Load image and return (original_rgb_uint8, input_tensor (1,3,H,W))."""
        if isinstance(image_input, np.ndarray):
            original = image_input
        else:
            path = Path(image_input)
            if path.suffix.lower() in (".dcm",):
                from utils.dicom_utils import load_dicom_as_array
                arr = load_dicom_as_array(path)
                # Convert to 3-channel RGB
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                original = arr
            else:
                pil_img = Image.open(path).convert("RGB")
                original = np.array(pil_img)

        # Ensure uint8 RGB
        if original.dtype != np.uint8:
            original = (
                ((original - original.min()) / (original.max() - original.min() + 1e-8)) * 255
            ).astype(np.uint8)
        if original.ndim == 2:
            original = np.stack([original] * 3, axis=-1)

        # PIL for TorchVision transforms
        pil = Image.fromarray(original)
        H, W = self.config.image_size
        tensor = TF.resize(pil, [H, W])
        tensor = TF.to_tensor(tensor)
        tensor = TF.normalize(tensor, list(self.config.mean), list(self.config.std))
        input_tensor = tensor.unsqueeze(0).to(self.device)

        return original, input_tensor

    def _run_segmentation(
        self,
        input_tensor: torch.Tensor,
        original_size: tuple[int, int],
    ) -> dict[str, Any]:
        """Run segmentation model and decode output."""
        assert self.seg_model is not None

        # Segmentation models expect single-channel or 3-channel input.
        # If the seg model was trained on grayscale, average channels.
        seg_input = input_tensor

        with torch.no_grad():
            ctx = torch.amp.autocast(device_type=self._device_type, enabled=self._amp)
            with ctx:
                seg_out = self.seg_model(seg_input)
                if isinstance(seg_out, list):
                    seg_out = seg_out[-1]

        return decode_segmentation(
            logits=seg_out,
            original_size=original_size,
            threshold=self.config.threshold,
            min_area_px=self.config.min_mask_area_px,
        )

    def _build_cam(self, method: str) -> Any:
        """Build the explainability module."""
        if method == "none":
            return None

        try:
            from models.explainability.gradcam import GradCAM, GradCAMPlusPlus
            from models.explainability.cam import CAM
            from models.explainability.attention_map import AttentionRollout

            if method == "gradcam":
                return GradCAM(self.classifier)
            elif method == "gradcam++":
                return GradCAMPlusPlus(self.classifier)
            elif method == "cam":
                return CAM(self.classifier)
            elif method == "rollout":
                return AttentionRollout(self.classifier)
            else:
                logger.warning("Unknown explainability method '%s', disabling.", method)
                return None
        except Exception as exc:
            logger.warning("Could not build CAM module (%s): %s", method, exc)
            return None

    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        classifier: nn.Module,
        segmentation_model: nn.Module | None = None,
        healthy_db: HealthyReferenceDatabase | None = None,
    ) -> "LungCarePipeline":
        """Convenience constructor from a :class:`PipelineConfig`."""
        return cls(
            classifier=classifier,
            config=config,
            segmentation_model=segmentation_model,
            healthy_db=healthy_db,
        )
