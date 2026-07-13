"""
tests/test_integration.py
──────────────────────────
End-to-end integration tests: dataset → model → trainer (mini-run).

All tests use synthetic data on CPU. One epoch, tiny batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from models import create_model
from training.losses import build_classification_loss
from training.trainer import EarlyStopping, Trainer, save_checkpoint, load_checkpoint
from utils.config import load_config, Config, TrainingConfig, DataConfig


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_cls_csv(tmp_path: "pytest.TempPathFactory", n: int = 8) -> str:
    """Create a synthetic classification CSV with real images."""
    import cv2
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    rows = []
    for i in range(n):
        p = img_dir / f"img_{i}.png"
        cv2.imwrite(str(p), (np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        rows.append({"image_path": str(p), "label": "Normal" if i % 2 == 0 else "TB",
                     "label_idx": i % 2})
    csv = tmp_path / "cls.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return str(csv)


def _mini_config() -> Config:
    cfg = Config()
    cfg.training.epochs      = 1
    cfg.training.batch_size  = 4
    cfg.training.lr          = 1e-3
    cfg.training.amp         = False   # CPU tests — AMP disabled
    cfg.training.grad_clip   = 1.0
    cfg.training.patience    = 999
    cfg.training.scheduler   = "cosine"
    cfg.training.checkpoint_dir = "checkpoints_test"
    cfg.training.log_dir        = "logs_test"
    cfg.data.num_classes    = 2
    cfg.data.class_names    = ["Normal", "TB"]
    cfg.data.image_size     = 64
    return cfg


# ─── EarlyStopping ────────────────────────────────────────────────────────────

class TestEarlyStopping:
    def test_not_triggered_on_improvement(self) -> None:
        es = EarlyStopping(patience=3, mode="max")
        for val in [0.1, 0.2, 0.3]:
            triggered = es(val)
        assert not triggered

    def test_triggered_after_patience(self) -> None:
        es = EarlyStopping(patience=2, mode="max")
        es(0.5)    # improvement — resets counter
        es(0.4)    # no improvement — counter=1
        es(0.3)    # no improvement — counter=2 → trigger
        assert es.triggered

    def test_min_mode(self) -> None:
        es = EarlyStopping(patience=2, mode="min")
        es(1.0)
        es(1.0)    # no improvement — counter=1
        triggered = es(1.0)    # counter=2
        assert triggered


# ─── Checkpoint ───────────────────────────────────────────────────────────────

class TestCheckpoint:
    def test_save_and_load(self, tmp_path: "pytest.TempPathFactory") -> None:
        model     = create_model("resnet50", num_classes=2, pretrained=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ckpt_path = tmp_path / "test.pth"

        save_checkpoint(ckpt_path, model, optimizer, epoch=5,
                        metrics={"val_f1": 0.88}, config_dict={})

        # Load into a fresh model
        new_model = create_model("resnet50", num_classes=2, pretrained=False)
        ckpt = load_checkpoint(ckpt_path, new_model)
        assert ckpt["epoch"] == 5
        assert ckpt["metrics"]["val_f1"] == 0.88


# ─── Full mini training loop ─────────────────────────────────────────────────

class TestTrainer:
    @pytest.fixture
    def loaders(self, tmp_path: "pytest.TempPathFactory") -> tuple[DataLoader, DataLoader]:
        from datasets.classification_dataset import ClassificationDataset
        csv = _make_cls_csv(tmp_path, n=8)
        ds  = ClassificationDataset(csv, split="train", image_size=64)
        dl  = DataLoader(ds, batch_size=4, shuffle=False)
        return dl, dl   # same for train and val in this test

    def test_one_epoch_completes(
        self, loaders: tuple[DataLoader, DataLoader], tmp_path: "pytest.TempPathFactory"
    ) -> None:
        train_dl, val_dl = loaders
        cfg     = _mini_config()
        cfg.training.checkpoint_dir = str(tmp_path / "ckpts")
        cfg.training.log_dir        = str(tmp_path / "logs")
        model   = create_model("resnet50", num_classes=2, pretrained=False)
        opt     = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = build_classification_loss("multiclass", label_smoothing=0.0)

        trainer = Trainer(
            model=model,
            train_loader=train_dl,
            val_loader=val_dl,
            criterion=loss_fn,
            optimizer=opt,
            config=cfg,
            task="multiclass",
            device="cpu",
            experiment_name="test_run",
        )
        result = trainer.train()

        assert "best_metric" in result
        assert "history"     in result
        assert len(result["history"]) == 1   # 1 epoch

    def test_best_checkpoint_saved(
        self, loaders: tuple[DataLoader, DataLoader], tmp_path: "pytest.TempPathFactory"
    ) -> None:
        train_dl, val_dl = loaders
        cfg = _mini_config()
        cfg.training.checkpoint_dir = str(tmp_path / "ckpts")
        cfg.training.log_dir        = str(tmp_path / "logs")
        model   = create_model("resnet50", num_classes=2, pretrained=False)
        opt     = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = build_classification_loss("multiclass", label_smoothing=0.0)

        trainer = Trainer(
            model=model, train_loader=train_dl, val_loader=val_dl,
            criterion=loss_fn, optimizer=opt, config=cfg,
            task="multiclass", device="cpu", experiment_name="ckpt_test",
        )
        result = trainer.train()

        import pathlib
        best = pathlib.Path(result["best_checkpoint"])
        assert best.exists()
