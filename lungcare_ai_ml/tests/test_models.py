"""
tests/test_models.py
─────────────────────
Unit tests for all four classifiers and the UNet segmentation model.

All tests use CPU + random weights (no disk I/O, no network).
"""
from __future__ import annotations

import pytest
import torch

from models import create_model
from models.resnet import ResNet50Classifier
from models.densenet import DenseNet121Classifier
from models.efficientnet import EfficientNetB0Classifier
from models.vit import ViTClassifier
from models.unet import UNet
from models.gradcam import GradCAM


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(params=["resnet50", "densenet121"])
def cnn_model(request: pytest.FixtureRequest) -> torch.nn.Module:
    """Parametrised fixture: returns one CNN classifier at a time."""
    return create_model(request.param, num_classes=6, pretrained=False)


@pytest.fixture
def resnet() -> ResNet50Classifier:
    return ResNet50Classifier(num_classes=2, pretrained=False)


@pytest.fixture
def dummy_batch() -> torch.Tensor:
    return torch.randn(2, 3, 224, 224)


# ─── Factory ─────────────────────────────────────────────────────────────────

class TestModelFactory:
    def test_create_resnet(self) -> None:
        m = create_model("resnet50", num_classes=2, pretrained=False)
        assert isinstance(m, ResNet50Classifier)

    def test_create_densenet(self) -> None:
        m = create_model("densenet121", num_classes=6, pretrained=False)
        assert isinstance(m, DenseNet121Classifier)

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown architecture"):
            create_model("mynet", num_classes=2)

    def test_all_registered(self) -> None:
        for name in ["resnet50", "densenet121", "efficientnet_b0"]:
            m = create_model(name, num_classes=2, pretrained=False)
            assert m is not None


# ─── ResNet50 ────────────────────────────────────────────────────────────────

class TestResNet50:
    def test_output_shape_binary(self, resnet: ResNet50Classifier, dummy_batch: torch.Tensor) -> None:
        out = resnet(dummy_batch)
        assert out.shape == (2, 2)

    def test_output_shape_6class(self, dummy_batch: torch.Tensor) -> None:
        m = ResNet50Classifier(num_classes=6, pretrained=False)
        out = m(dummy_batch)
        assert out.shape == (2, 6)

    def test_get_target_layer(self, resnet: ResNet50Classifier) -> None:
        layer = resnet.get_target_layer()
        assert layer is resnet.layer4

    def test_get_features_shape(self, resnet: ResNet50Classifier, dummy_batch: torch.Tensor) -> None:
        feats = resnet.get_features(dummy_batch)
        assert feats.ndim == 4         # (B, C, h, w)
        assert feats.shape[0] == 2


# ─── DenseNet121 ─────────────────────────────────────────────────────────────

class TestDenseNet121:
    def test_output_shape(self, dummy_batch: torch.Tensor) -> None:
        m = DenseNet121Classifier(num_classes=6, pretrained=False)
        out = m(dummy_batch)
        assert out.shape == (2, 6)

    def test_get_target_layer(self) -> None:
        m = DenseNet121Classifier(num_classes=2, pretrained=False)
        assert m.get_target_layer() is m.features.denseblock4


# ─── UNet ────────────────────────────────────────────────────────────────────

class TestUNet:
    def test_output_shape_binary(self) -> None:
        model = UNet(in_channels=3, out_channels=1)
        x     = torch.randn(2, 3, 224, 224)
        out   = model(x)
        assert out.shape == (2, 1, 224, 224)

    def test_count_parameters(self) -> None:
        model = UNet()
        n = model.count_parameters()
        assert n > 0


# ─── GradCAM ─────────────────────────────────────────────────────────────────

class TestGradCAM:
    def test_heatmap_shape(self, resnet: ResNet50Classifier) -> None:
        x = torch.randn(1, 3, 224, 224)
        with GradCAM(resnet, device="cpu") as cam:
            heatmap, cls_idx = cam.compute(x, output_size=(224, 224))
        assert heatmap.shape == (224, 224)
        assert 0.0 <= heatmap.min()
        assert heatmap.max() <= 1.0
        assert isinstance(cls_idx, int)

    def test_overlay_shape(self) -> None:
        import numpy as np
        img  = (np.random.rand(224, 224, 3) * 255).astype("uint8")
        heat = np.random.rand(224, 224).astype("float32")
        out  = GradCAM.overlay(img, heat)
        assert out.shape == (224, 224, 3)
        assert out.dtype.name == "uint8"

    def test_none_target_layer_raises(self) -> None:
        """ViT's get_target_layer() returns None → GradCAM should raise."""
        m = ViTClassifier(num_classes=2, pretrained=False)
        with pytest.raises(ValueError, match="None from get_target_layer"):
            _ = GradCAM(m)


# ─── CNN parametrised ────────────────────────────────────────────────────────

class TestCNNOutput:
    def test_forward_pass(
        self, cnn_model: torch.nn.Module, dummy_batch: torch.Tensor
    ) -> None:
        out = cnn_model(dummy_batch)
        assert out.shape[0] == 2
        assert out.shape[1] == 6

    def test_no_nan_in_output(
        self, cnn_model: torch.nn.Module, dummy_batch: torch.Tensor
    ) -> None:
        out = cnn_model(dummy_batch)
        assert not torch.isnan(out).any()
