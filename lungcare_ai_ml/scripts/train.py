"""
Classification training entry point for LungCare AI.

Reads configuration from a YAML file and optional CLI overrides,
then runs the full training + validation loop.

Usage
-----
  # Basic — ResNet50, 100 epochs, auto-detect GPU
  python scripts/train_classifier.py \\
      --config configs/classification_config.yaml \\
      --model resnet50 \\
      --data-dir data/prepared/classification \\
      --experiment tb_resnet50

  # AMP + gradient accumulation + focal loss
  python scripts/train_classifier.py \\
      --model efficientnet_b0 --amp --grad-accum 4 --loss focal \\
      --experiment tb_efficientnet_focal

  # Multi-GPU DataParallel
  python scripts/train_classifier.py \\
      --model densenet121 --data-parallel \\
      --experiment tb_densenet121_dp

  # Resume from checkpoint
  python scripts/train_classifier.py \\
      --model resnet50 \\
      --resume checkpoints/tb_resnet50/best.pth \\
      --experiment tb_resnet50_resumed

  # Dry-run (verify dataset loading and model construction, no training)
  python scripts/train_classifier.py --model resnet50 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("lungcare.scripts.train_classifier")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LungCare AI — Classification Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Config ─────────────────────────────────────────────────────────────
    parser.add_argument("--config", type=Path, default=Path("configs/classification_config.yaml"))
    parser.add_argument("--experiment", type=str, default="experiment", help="Experiment name.")

    # ── Data ───────────────────────────────────────────────────────────────
    parser.add_argument("--data-dir", type=Path, default=Path("data/prepared/classification"))
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cache", action="store_true", help="Cache datasets in RAM.")

    # ── Model ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        choices=["resnet50", "densenet121", "efficientnet_b0", "vit_b16"],
        default="resnet50",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass", "multilabel"],
        default="multiclass",
    )
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)

    # ── Training ───────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--optimizer", choices=["adamw", "adam", "sgd", "rmsprop"], default="adamw"
    )
    parser.add_argument(
        "--scheduler",
        choices=["warmup_cosine", "cosine", "onecycle", "step", "reduce_on_plateau"],
        default="warmup_cosine",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5)

    # ── Loss ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--loss",
        choices=["cross_entropy", "focal", "label_smoothing", "bce"],
        default="cross_entropy",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # ── Advanced ───────────────────────────────────────────────────────────
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1, dest="grad_accum_steps")
    parser.add_argument("--data-parallel", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--resume", type=Path, default=None)

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--device", type=str, default=None, help="'cpu', 'cuda', 'cuda:0' …")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    parser.add_argument("--dry-run", action="store_true", help="Build everything but skip training.")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Logging ────────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                Path(args.log_dir) / args.experiment / "train.log",
                mode="a",
                delay=True,
            ),
        ],
    )

    import torch
    from torch.utils.data import DataLoader

    # ── Seed ───────────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.seed import set_seed
    set_seed(args.seed)

    # ── Device ─────────────────────────────────────────────────────────────────
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    # ── Dataset ────────────────────────────────────────────────────────────────
    from datasets.classification_dataset import ClassificationDataset

    data_dir = args.data_dir.resolve()
    train_csv = args.train_csv or data_dir / "train.csv"
    val_csv = args.val_csv or data_dir / "val.csv"

    class_names: list[str] = [
        "Healthy", "Tuberculosis", "Pneumonia",
        "COVID-19", "Lung Cancer", "Pulmonary Fibrosis",
    ]
    img_size = tuple(args.image_size) if args.image_size else (224, 224)

    train_ds = ClassificationDataset(
        csv_path=train_csv,
        image_size=img_size,
        split="train",
        num_classes=args.num_classes,
        task=args.task,
        use_cache=args.cache,
        class_names=class_names,
    )
    val_ds = ClassificationDataset(
        csv_path=val_csv,
        image_size=img_size,
        split="val",
        num_classes=args.num_classes,
        task=args.task,
        use_cache=False,
        class_names=class_names,
    )

    logger.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory="cuda" in device,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory="cuda" in device,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    from models import create_classifier
    model = create_classifier(
        name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout,
    )
    if args.freeze_backbone:
        model.freeze_backbone()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s | Trainable params: %s", args.model, f"{n_params:,}")

    if args.dry_run:
        logger.info("Dry-run complete — exiting before training.")
        return

    # ── Trainer config ────────────────────────────────────────────────────────
    from training.classification_trainer import ClassificationConfig, ClassificationTrainer
    from training.trainer import EarlyStoppingConfig, OptimizerConfig, SchedulerConfig

    trainer_cfg = ClassificationConfig(
        task=args.task,
        num_classes=args.num_classes,
        epochs=args.epochs,
        amp=args.amp,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps,
        data_parallel=args.data_parallel,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment,
        monitor_metric="val_f1_macro" if args.task == "multiclass" else "val_f1",
        monitor_mode="max",
        resume_from=args.resume,
        loss_type=args.loss,
        label_smoothing=args.label_smoothing,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
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
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        device=device,
        class_names=class_names,
    )
    trainer.train()
    trainer.log_per_class_f1(epoch=trainer_cfg.epochs)

    # ── Final evaluation on val set ───────────────────────────────────────────
    from evaluation.evaluator import Evaluator
    best_ckpt = args.checkpoint_dir / args.experiment / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded best checkpoint for final evaluation.")

    evaluator = Evaluator(
        model=model,
        loader=val_loader,
        task=args.task,
        num_classes=args.num_classes,
        device=device,
        class_names=class_names,
        amp=args.amp,
    )
    result = evaluator.evaluate()
    evaluator.save_predictions(
        result,
        output_path=args.log_dir / args.experiment / "final_val_predictions.json",
    )
    logger.info("Final val metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
