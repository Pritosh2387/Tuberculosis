"""
Segmentation training entry point for LungCare AI.

Supports U-Net, Attention U-Net, and U-Net++ (with deep supervision).

Usage
-----
  # Standard U-Net — BCE+Dice loss
  python scripts/train_segmentation.py \\
      --model unet \\
      --data-dir data/prepared/segmentation \\
      --experiment seg_unet

  # Attention U-Net — focal+dice
  python scripts/train_segmentation.py \\
      --model attention_unet --loss focal_dice \\
      --experiment seg_attn_unet

  # U-Net++ with deep supervision
  python scripts/train_segmentation.py \\
      --model unet_plus_plus --deep-supervision \\
      --ds-weights 0.2 0.3 0.5 \\
      --experiment seg_unetpp

  # CT scans (1-channel DICOM input)
  python scripts/train_segmentation.py \\
      --model unet --in-channels 1 \\
      --data-dir data/prepared/ct_segmentation \\
      --experiment seg_ct_unet

  # Resume
  python scripts/train_segmentation.py \\
      --model unet \\
      --resume checkpoints/seg_unet/best.pth \\
      --experiment seg_unet_resumed
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("lungcare.scripts.train_segmentation")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LungCare AI — Segmentation Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Data ───────────────────────────────────────────────────────────────
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared/segmentation"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache", action="store_true")

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        choices=["unet", "attention_unet", "unet_plus_plus"],
        default="unet",
    )
    parser.add_argument("--in-channels", type=int, default=3,
                        help="1 for grayscale/CT, 3 for RGB CXR.")
    parser.add_argument("--out-channels", type=int, default=1,
                        help="1 for binary segmentation.")
    parser.add_argument("--features", type=int, nargs="+",
                        default=[64, 128, 256, 512],
                        help="Encoder channel counts.")
    parser.add_argument("--bilinear", action="store_true", default=True)
    parser.add_argument("--no-bilinear", dest="bilinear", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.2)

    # ── Deep supervision (U-Net++) ─────────────────────────────────────────
    parser.add_argument("--deep-supervision", action="store_true")
    parser.add_argument("--ds-weights", type=float, nargs="+", default=None,
                        help="Per-level loss weights for deep supervision (coarse→fine).")

    # ── Loss ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--loss", choices=["dice", "bce", "bce_dice", "focal_dice"],
        default="bce_dice",
    )
    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5)

    # ── Training ───────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    parser.add_argument(
        "--scheduler",
        choices=["warmup_cosine", "cosine", "reduce_on_plateau", "step"],
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1, dest="grad_accum_steps")
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--mask-log-interval", type=int, default=10,
                        help="Log mask grid to TensorBoard every N epochs.")

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--experiment", type=str, default="seg_experiment")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(args.log_dir) / args.experiment / "train_seg.log",
                mode="a", delay=True,
            ),
        ],
    )

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.seed import set_seed
    set_seed(args.seed)

    # ── Device ─────────────────────────────────────────────────────────────────
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Dataset ────────────────────────────────────────────────────────────────
    from datasets.segmentation_dataset import SegmentationDataset

    data_dir = args.data_dir.resolve()
    train_csv = args.train_csv or data_dir / "train.csv"
    val_csv = args.val_csv or data_dir / "val.csv"
    img_size = tuple(args.image_size)

    train_ds = SegmentationDataset(
        csv_path=train_csv,
        image_size=img_size,
        split="train",
        in_channels=args.in_channels,
        use_cache=args.cache,
    )
    val_ds = SegmentationDataset(
        csv_path=val_csv,
        image_size=img_size,
        split="val",
        in_channels=args.in_channels,
        use_cache=False,
    )

    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory="cuda" in device,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory="cuda" in device,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from models import create_segmentation_model

    model_kwargs: dict = dict(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        features=tuple(args.features),
        bilinear=args.bilinear,
        dropout_rate=args.dropout,
    )
    if args.model == "unet_plus_plus":
        model_kwargs["deep_supervision"] = args.deep_supervision

    model = create_segmentation_model(args.model, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | Params: %s", args.model, f"{n_params:,}")

    if args.dry_run:
        logger.info("Dry-run complete — exiting before training.")
        return

    # ── Trainer config ────────────────────────────────────────────────────────
    from training.segmentation_trainer import SegmentationConfig, SegmentationTrainer
    from training.trainer import EarlyStoppingConfig, OptimizerConfig, SchedulerConfig

    trainer_cfg = SegmentationConfig(
        task="segmentation",
        num_classes=args.out_channels,
        epochs=args.epochs,
        amp=args.amp,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        data_parallel=args.data_parallel,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment,
        monitor_metric="val_dice",
        monitor_mode="max",
        resume_from=args.resume,
        loss_type=args.loss,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        deep_supervision=args.deep_supervision,
        ds_weights=args.ds_weights,
        threshold=args.threshold,
        mask_log_interval=args.mask_log_interval,
        optimizer=OptimizerConfig(
            name=args.optimizer, lr=args.lr, weight_decay=args.weight_decay
        ),
        scheduler=SchedulerConfig(
            name=args.scheduler,
            T_max=args.epochs,
            warmup_epochs=args.warmup_epochs,
        ),
        early_stopping=EarlyStoppingConfig(
            enabled=True,
            patience=args.early_stopping_patience,
            mode="max",
        ),
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        device=device,
    )

    # Register post-epoch mask logging hook
    original_post = trainer._post_train_epoch
    def _patched_post(epoch: int) -> None:
        original_post(epoch)
        trainer.log_mask_grid(epoch)

    trainer._post_train_epoch = _patched_post  # type: ignore[method-assign]
    trainer.train()

    logger.info("Segmentation training complete. Checkpoints at: %s",
                args.checkpoint_dir / args.experiment)


if __name__ == "__main__":
    main()
