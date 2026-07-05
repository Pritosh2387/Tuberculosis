"""
Tests for the inference pipeline (inference/pipeline.py and postprocessing.py).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def dummy_classifier():
    """Returns a random-weight ResNet50 classifier."""
    from models.classification.resnet import ResNet50Classifier
    return ResNet50Classifier(num_classes=6, pretrained=False).eval()


@pytest.fixture()
def dummy_seg_model():
    """Returns a tiny random-weight U-Net for testing."""
    from models.segmentation.unet import UNet
    return UNet(in_channels=3, out_channels=1, features=(8, 16)).eval()


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    """Save a synthetic RGB image to a temporary file."""
    img_arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img_path = tmp_path / "test_image.png"
    Image.fromarray(img_arr).save(img_path)
    return img_path


@pytest.fixture()
def sample_image_np() -> np.ndarray:
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


# ─── Postprocessing ───────────────────────────────────────────────────────────


class TestDecodeClassification:
    def test_multiclass_output_keys(self) -> None:
        from inference.postprocessing import decode_classification

        logits = torch.randn(1, 6)
        classes = ["Healthy", "TB", "Pneumonia", "COVID-19", "Cancer", "Fibrosis"]
        result = decode_classification(logits, class_names=classes, task="multiclass")
        assert "pred_class" in result
        assert "confidence" in result
        assert "all_probs" in result
        assert "top_k" in result
        assert "is_abnormal" in result

    def test_binary_output(self) -> None:
        from inference.postprocessing import decode_classification

        logits = torch.tensor([[5.0]])  # Strong positive → abnormal
        result = decode_classification(logits, class_names=["Healthy", "Abnormal"],
                                       task="binary", threshold=0.5)
        assert result["pred_class"] == "Abnormal"
        assert result["is_abnormal"] is True

    def test_confidence_in_range(self) -> None:
        from inference.postprocessing import decode_classification

        logits = torch.randn(1, 6)
        classes = ["C0", "C1", "C2", "C3", "C4", "C5"]
        result = decode_classification(logits, class_names=classes, task="multiclass")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_top_k_length(self) -> None:
        from inference.postprocessing import decode_classification

        logits = torch.randn(1, 6)
        classes = [f"C{i}" for i in range(6)]
        result = decode_classification(logits, class_names=classes, task="multiclass", top_k=3)
        assert len(result["top_k"]) == 3


class TestDecodeSegmentation:
    def test_output_keys(self) -> None:
        from inference.postprocessing import decode_segmentation

        logits = torch.randn(1, 1, 64, 64)
        result = decode_segmentation(logits, original_size=(256, 256))
        assert "mask" in result
        assert "bboxes" in result
        assert "num_regions" in result
        assert "coverage_pct" in result
        assert "prob_map" in result

    def test_mask_shape(self) -> None:
        from inference.postprocessing import decode_segmentation

        logits = torch.randn(1, 1, 64, 64)
        result = decode_segmentation(logits, original_size=(512, 512))
        assert result["mask"].shape == (512, 512)

    def test_mask_binary_values(self) -> None:
        from inference.postprocessing import decode_segmentation

        logits = torch.randn(1, 1, 64, 64)
        result = decode_segmentation(logits, original_size=(128, 128))
        unique = np.unique(result["mask"])
        assert set(unique.tolist()).issubset({0, 255})

    def test_strong_positive_logit_fills_mask(self) -> None:
        from inference.postprocessing import decode_segmentation

        logits = torch.full((1, 1, 64, 64), 10.0)  # All foreground
        result = decode_segmentation(logits, original_size=(128, 128),
                                     threshold=0.5, min_area_px=1)
        assert result["coverage_pct"] > 50.0


class TestOverlayFunctions:
    def test_overlay_heatmap_shape(self, sample_image_np: np.ndarray) -> None:
        from inference.postprocessing import overlay_heatmap

        heatmap = np.random.rand(256, 256).astype(np.float32)
        result = overlay_heatmap(sample_image_np, heatmap)
        assert result.shape == sample_image_np.shape
        assert result.dtype == np.uint8

    def test_overlay_mask_shape(self, sample_image_np: np.ndarray) -> None:
        from inference.postprocessing import overlay_mask

        mask = (np.random.rand(256, 256) > 0.5).astype(np.uint8) * 255
        result = overlay_mask(sample_image_np, mask)
        assert result.shape == sample_image_np.shape


# ─── Pipeline ─────────────────────────────────────────────────────────────────


class TestLungCarePipeline:
    @pytest.fixture()
    def pipeline(self, dummy_classifier):
        from inference.pipeline import LungCarePipeline, PipelineConfig

        cfg = PipelineConfig(
            class_names=["Healthy", "TB", "Pneumonia", "COVID-19", "Cancer", "Fibrosis"],
            task="multiclass",
            image_size=(64, 64),
            explainability_method="none",
            run_segmentation=False,
            amp=False,
            device="cpu",
        )
        return LungCarePipeline(classifier=dummy_classifier, config=cfg)

    def test_predict_from_array(self, pipeline, sample_image_np: np.ndarray) -> None:
        result = pipeline.predict(sample_image_np, case_id="test_001")
        assert result.report is not None
        assert "prediction" in result.report
        assert "confidence" in result.report
        assert "findings" in result.report

    def test_predict_from_file(
        self, pipeline, sample_image_path: Path
    ) -> None:
        result = pipeline.predict(sample_image_path, case_id="test_file")
        assert result.classification is not None
        assert result.inference_time_s > 0

    def test_report_schema(self, pipeline, sample_image_np: np.ndarray) -> None:
        result = pipeline.predict(sample_image_np)
        r = result.report
        for key in ("status", "prediction", "confidence", "all_scores",
                    "findings", "localization", "segmentation", "timestamp"):
            assert key in r, f"Missing key: {key}"

    def test_save_result(
        self, pipeline, sample_image_np: np.ndarray, tmp_path: Path
    ) -> None:
        result = pipeline.predict(sample_image_np, case_id="save_test")
        saved = pipeline.save_result(result, output_dir=tmp_path, case_id="save_test")
        assert "report_json" in saved
        assert saved["report_json"].exists()
        assert "report_md" in saved
        assert saved["report_md"].exists()

    def test_with_segmentation(
        self, dummy_classifier, dummy_seg_model, sample_image_np: np.ndarray
    ) -> None:
        from inference.pipeline import LungCarePipeline, PipelineConfig

        cfg = PipelineConfig(
            class_names=["Healthy", "TB", "Pneumonia", "COVID-19", "Cancer", "Fibrosis"],
            task="multiclass",
            image_size=(64, 64),
            explainability_method="none",
            run_segmentation=True,
            amp=False,
            device="cpu",
        )
        pl = LungCarePipeline(
            classifier=dummy_classifier,
            config=cfg,
            segmentation_model=dummy_seg_model,
        )
        result = pl.predict(sample_image_np)
        assert result.segmentation is not None
        assert "mask" in result.segmentation
