"""
tests/test_inference.py
────────────────────────
Tests for the inference pipeline, decode_classification, and ReportGenerator.

All tests use CPU + random weights. No real images required.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from evaluation.report_generator import ReportGenerator
from inference.pipeline import LungCarePipeline, PipelineResult, decode_classification
from models.resnet import ResNet50Classifier


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def classifier() -> ResNet50Classifier:
    return ResNet50Classifier(num_classes=2, pretrained=False).eval()


@pytest.fixture
def pipeline(classifier: ResNet50Classifier) -> LungCarePipeline:
    return LungCarePipeline(
        classifier=classifier,
        class_names=["Normal", "Tuberculosis"],
        image_size=64,
        device="cpu",
        explainability="gradcam",
    )


@pytest.fixture
def dummy_image() -> np.ndarray:
    return (np.random.rand(128, 128, 3) * 255).astype(np.uint8)


# ─── decode_classification ───────────────────────────────────────────────────

class TestDecodeClassification:
    def test_multiclass_keys(self) -> None:
        logits = torch.randn(1, 2)
        result = decode_classification(logits, ["Normal", "TB"], task="multiclass")
        assert "pred_class"  in result
        assert "confidence"  in result
        assert "all_probs"   in result
        assert "top_k_preds" in result

    def test_probs_sum_to_one(self) -> None:
        logits = torch.randn(1, 6)
        result = decode_classification(logits, [str(i) for i in range(6)])
        total  = sum(result["all_probs"].values())
        assert abs(total - 1.0) < 1e-4

    def test_pred_idx_in_range(self) -> None:
        logits = torch.randn(1, 4)
        result = decode_classification(logits, ["A", "B", "C", "D"])
        assert 0 <= result["pred_idx"] < 4

    def test_confidence_in_range(self) -> None:
        logits = torch.randn(1, 2)
        result = decode_classification(logits, ["Normal", "TB"])
        assert 0.0 <= result["confidence"] <= 1.0


# ─── LungCarePipeline ────────────────────────────────────────────────────────

class TestLungCarePipeline:
    def test_predict_returns_result(
        self, pipeline: LungCarePipeline, dummy_image: np.ndarray
    ) -> None:
        result = pipeline.predict(dummy_image)
        assert isinstance(result, PipelineResult)

    def test_result_has_report(
        self, pipeline: LungCarePipeline, dummy_image: np.ndarray
    ) -> None:
        result = pipeline.predict(dummy_image)
        report = result.report
        for key in ("prediction", "confidence", "all_scores", "status", "findings"):
            assert key in report, f"Missing key: {key}"

    def test_heatmap_shape(
        self, pipeline: LungCarePipeline, dummy_image: np.ndarray
    ) -> None:
        result = pipeline.predict(dummy_image)
        if result.heatmap is not None:
            h, w = dummy_image.shape[:2]
            assert result.heatmap.shape == (h, w)

    def test_overlay_shape(
        self, pipeline: LungCarePipeline, dummy_image: np.ndarray
    ) -> None:
        result = pipeline.predict(dummy_image)
        if result.overlay_image is not None:
            assert result.overlay_image.shape == dummy_image.shape

    def test_confidence_is_float(
        self, pipeline: LungCarePipeline, dummy_image: np.ndarray
    ) -> None:
        result = pipeline.predict(dummy_image)
        assert isinstance(result.report["confidence"], float)

    def test_inference_time_recorded(
        self, pipeline: LungCarePipeline, dummy_image: np.ndarray
    ) -> None:
        result = pipeline.predict(dummy_image)
        assert result.inference_time_s > 0.0

    def test_no_gradcam_for_vit(self, dummy_image: np.ndarray) -> None:
        """ViT with explainability='gradcam' should fall back gracefully."""
        from models.vit import ViTClassifier
        vit = ViTClassifier(num_classes=2, pretrained=False).eval()
        p   = LungCarePipeline(
            classifier=vit,
            class_names=["Normal", "TB"],
            image_size=224,
            explainability="gradcam",
        )
        result = p.predict(dummy_image)
        # Should not crash; heatmap is None because target_layer is None
        assert isinstance(result, PipelineResult)


# ─── ReportGenerator ─────────────────────────────────────────────────────────

class TestReportGenerator:
    @pytest.fixture
    def gen(self) -> ReportGenerator:
        return ReportGenerator(
            class_names=["Normal", "Tuberculosis"],
            model_version="v2.0",
            finding_threshold=0.5,
        )

    def test_generate_schema(self, gen: ReportGenerator) -> None:
        probs  = np.array([0.1, 0.9])
        report = gen.generate(probs)
        required = {
            "prediction", "confidence", "all_scores",
            "status", "findings", "localization",
            "timestamp", "model_version",
        }
        assert required.issubset(report.keys())

    def test_abnormal_status(self, gen: ReportGenerator) -> None:
        probs  = np.array([0.1, 0.9])
        report = gen.generate(probs)
        assert report["status"] == "abnormal"
        assert report["prediction"] == "Tuberculosis"

    def test_normal_status(self, gen: ReportGenerator) -> None:
        probs  = np.array([0.95, 0.05])
        report = gen.generate(probs)
        assert report["status"] == "normal"

    def test_with_heatmap(self, gen: ReportGenerator) -> None:
        probs   = np.array([0.1, 0.9])
        heatmap = np.random.rand(224, 224).astype(np.float32)
        report  = gen.generate(probs, heatmap=heatmap)
        assert report["localization"]["method"] == "GradCAM"

    def test_markdown_output(self, gen: ReportGenerator) -> None:
        probs  = np.array([0.3, 0.7])
        report = gen.generate(probs)
        md     = gen.to_markdown(report)
        assert "LungCare AI" in md
        assert "Tuberculosis" in md

    def test_save_json(self, gen: ReportGenerator, tmp_path: "pytest.TempPathFactory") -> None:
        probs  = np.array([0.2, 0.8])
        report = gen.generate(probs, case_id="test_001")
        path   = gen.save_report(report, tmp_path / "report.json")
        assert path.exists()

        import json
        loaded = json.loads(path.read_text())
        assert loaded["case_id"] == "test_001"
