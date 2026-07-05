"""
Tests for all model architectures — classification and segmentation.

Uses randomly initialised weights (pretrained=False) and synthetic tensors
so tests run fast on CPU with no internet access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


NUM_CLASSES = 6
BATCH = 2
IMG_SIZE = 224


# ─── Classification models ────────────────────────────────────────────────────


class TestResNet50:
    @pytest.fixture()
    def model(self):
        from models.classification.resnet import ResNet50Classifier
        return ResNet50Classifier(num_classes=NUM_CLASSES, pretrained=False)

    def test_forward_shape(self, model):
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_get_features_shape(self, model):
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        feats = model.get_features(x)
        assert feats.ndim == 4
        assert feats.shape[0] == 1

    def test_get_target_layer(self, model):
        layer = model.get_target_layer()
        assert layer is not None

    def test_predict_dict(self, model):
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        result = model.predict(x)
        assert "pred_class" in result
        assert "confidence" in result
        assert "probabilities" in result

    def test_freeze_backbone(self, model):
        model.freeze_backbone()
        frozen = [p for name, p in model.named_parameters()
                  if "classifier" not in name and not p.requires_grad]
        assert len(frozen) > 0


class TestDenseNet121:
    @pytest.fixture()
    def model(self):
        from models.classification.densenet import DenseNet121Classifier
        return DenseNet121Classifier(num_classes=NUM_CLASSES, pretrained=False)

    def test_forward_shape(self, model):
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_get_features_returns_4d(self, model):
        x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        feats = model.get_features(x)
        assert feats.ndim == 4


class TestEfficientNetB0:
    @pytest.fixture()
    def model(self):
        from models.classification.efficientnet import EfficientNetB0Classifier
        return EfficientNetB0Classifier(num_classes=NUM_CLASSES, pretrained=False)

    def test_forward_shape(self, model):
        x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)


class TestViTClassifier:
    @pytest.fixture()
    def model(self):
        from models.classification.vit import ViTClassifier
        return ViTClassifier(num_classes=NUM_CLASSES, pretrained=False)

    def test_forward_shape(self, model):
        x = torch.randn(BATCH, 3, 224, 224)
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)


# ─── Factory ─────────────────────────────────────────────────────────────────


class TestModelFactory:
    @pytest.mark.parametrize("arch", [
        "resnet50", "densenet121", "efficientnet_b0", "vit_b16"
    ])
    def test_create_classifier(self, arch: str):
        from models import create_classifier
        model = create_classifier(arch, num_classes=NUM_CLASSES, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, NUM_CLASSES)

    @pytest.mark.parametrize("arch", ["unet", "attention_unet", "unet_plus_plus"])
    def test_create_segmentation_model(self, arch: str):
        from models import create_segmentation_model
        kwargs = dict(in_channels=1, out_channels=1, features=(16, 32, 64))
        if arch == "unet_plus_plus":
            kwargs["deep_supervision"] = False
        model = create_segmentation_model(arch, **kwargs)
        x = torch.randn(1, 1, 128, 128)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape[-2:] == (128, 128)


# ─── Segmentation models ─────────────────────────────────────────────────────


class TestUNet:
    @pytest.fixture()
    def model(self):
        from models.segmentation.unet import UNet
        return UNet(in_channels=1, out_channels=1, features=(16, 32, 64), bilinear=True)

    def test_forward_preserves_spatial(self, model):
        x = torch.randn(BATCH, 1, 128, 128)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, 1, 128, 128)

    def test_count_parameters(self, model):
        assert model.count_parameters() > 0

    def test_odd_spatial_dims(self, model):
        """U-Net must handle odd H/W via padding."""
        x = torch.randn(1, 1, 129, 133)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape[2:] == (129, 133)


class TestAttentionUNet:
    @pytest.fixture()
    def model(self):
        from models.segmentation.attention_unet import AttentionUNet
        return AttentionUNet(in_channels=1, out_channels=1, features=(16, 32, 64))

    def test_forward_shape(self, model):
        x = torch.randn(BATCH, 1, 128, 128)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, 1, 128, 128)

    def test_get_attention_maps(self, model):
        x = torch.randn(1, 1, 128, 128)
        maps = model.get_attention_maps(x)
        assert len(maps) > 0
        for m in maps:
            assert m.shape[1] == 1  # Single attention channel


class TestUNetPlusPlus:
    @pytest.fixture()
    def model_ds(self):
        from models.segmentation.unet_plus_plus import UNetPlusPlus
        return UNetPlusPlus(
            in_channels=1, out_channels=1,
            features=(16, 32, 64, 128), deep_supervision=True
        )

    @pytest.fixture()
    def model_no_ds(self):
        from models.segmentation.unet_plus_plus import UNetPlusPlus
        return UNetPlusPlus(
            in_channels=1, out_channels=1,
            features=(16, 32, 64), deep_supervision=False
        )

    def test_train_returns_list(self, model_ds):
        model_ds.train()
        x = torch.randn(BATCH, 1, 128, 128)
        out = model_ds(x)
        assert isinstance(out, list)
        assert len(out) > 0

    def test_eval_returns_tensor(self, model_ds):
        model_ds.eval()
        x = torch.randn(BATCH, 1, 128, 128)
        with torch.no_grad():
            out = model_ds(x)
        assert isinstance(out, torch.Tensor)

    def test_no_ds_always_tensor(self, model_no_ds):
        for mode in [True, False]:
            model_no_ds.train(mode)
            x = torch.randn(1, 1, 64, 64)
            if not mode:
                with torch.no_grad():
                    out = model_no_ds(x)
            else:
                out = model_no_ds(x)
            assert isinstance(out, torch.Tensor)
