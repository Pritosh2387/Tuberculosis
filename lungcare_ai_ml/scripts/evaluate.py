"""
Model evaluation script for LungCare AI.

Loads a trained checkpoint, runs a full evaluation pass over a test set,
and saves:
  - Scalar metrics JSON
  - Per-sample predictions JSON
  - Confusion matrix PNG (multiclass only)
  - Markdown summary report

Usage
-----
  # Evaluate classification model
  python scripts/evaluate.py \\
      --checkpoint checkpoints/tb_resnet50/best.pth \\
      --data-dir data/prepared/classification \\
      --split test \\
      --task multiclass \\
      --model resnet50 \\
      --output results/tb_resnet50

  # Evaluate segmentation model
  python scripts/evaluate.py \\
      --checkpoint checkpoints/seg_unet/best.pth \\
      --data-dir data/prepared/segmentation \\
      --task segmentation \\
      --model unet \\
      --output results/seg_unet

  # Evaluate on a custom CSV
  python scripts/evaluate.py \\
      --checkpoint checkpoints/tb_resnet50/best.pth \\
      --csv data/custom_test.csv \\
      --task multiclass \\
      --model resnet50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("lungcare.scripts.evaluate")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LungCare AI — Model Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint (.pth).")
    parser.add_argument("--model",
                        choices=["resnet50", "densenet121", "efficientnet_b0",
                                 "vit_b16", "unet", "attention_unet", "unet_plus_plus"],
                        required=True)
    parser.add_argument("--task",
                        choices=["binary", "multiclass", "multilabel", "segmentation"],
                        default="multiclass")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--in-channels", type=int, default=3)

    # ── Data ───────────────────────────────────────────────────────────────
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None,
                        help="Direct path to evaluation CSV.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output", type=Path, default=Path("results/eval"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])

    return parser.parse_args()


def _plot_confusion_matrix(
    cm: "np.ndarray",
    class_names: list[str],
    output_path: Path,
) -> None:
    """Save confusion matrix as a PNG heatmap."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=8,
                )

        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_title("Confusion Matrix", fontsize=13, pad=12)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", output_path)
    except ImportError:
        logger.warning("matplotlib not installed — skipping confusion matrix plot.")


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).parent.parent))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    class_names: list[str] = [
        "Healthy", "Tuberculosis", "Pneumonia",
        "COVID-19", "Lung Cancer", "Pulmonary Fibrosis",
    ]

    # ── Resolve CSV ────────────────────────────────────────────────────────────
    if args.csv:
        eval_csv = args.csv.resolve()
    elif args.data_dir:
        eval_csv = (args.data_dir / f"{args.split}.csv").resolve()
    else:
        logger.error("Provide --data-dir or --csv.")
        sys.exit(1)

    if not eval_csv.exists():
        logger.error("CSV not found: %s", eval_csv)
        sys.exit(1)

    # ── Dataset ────────────────────────────────────────────────────────────────
    img_size = tuple(args.image_size)

    if args.task == "segmentation":
        from datasets.segmentation_dataset import SegmentationDataset
        ds = SegmentationDataset(
            csv_path=eval_csv, image_size=img_size, split="val",
            in_channels=args.in_channels, use_cache=False,
        )
    else:
        from datasets.classification_dataset import ClassificationDataset
        ds = ClassificationDataset(
            csv_path=eval_csv, image_size=img_size, split="val",
            num_classes=args.num_classes, task=args.task, use_cache=False,
            class_names=class_names,
        )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory="cuda" in device,
    )
    logger.info("Evaluation set: %d samples", len(ds))

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # ── Build model ────────────────────────────────────────────────────────────
    if args.task == "segmentation":
        from models import create_segmentation_model
        model = create_segmentation_model(
            args.model,
            in_channels=args.in_channels,
            out_channels=args.num_classes,
        )
    else:
        from models import create_classifier
        model = create_classifier(
            args.model,
            num_classes=args.num_classes,
            pretrained=False,
        )

    model.load_state_dict(state["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d.", state.get("epoch", "?"))

    # ── Evaluate ───────────────────────────────────────────────────────────────
    from evaluation.evaluator import Evaluator

    evaluator = Evaluator(
        model=model,
        loader=loader,
        task=args.task,
        num_classes=args.num_classes,
        device=device,
        class_names=class_names,
        threshold=args.threshold,
        amp=args.amp,
    )
    result = evaluator.evaluate()

    # ── Save outputs ───────────────────────────────────────────────────────────
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metrics JSON
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result.metrics, f, indent=2)
    logger.info("Metrics saved → %s", metrics_path)

    # Markdown summary
    md_lines = [
        f"# LungCare AI — Evaluation Report",
        f"",
        f"**Checkpoint**: `{ckpt_path}`  ",
        f"**Model**: `{args.model}`  ",
        f"**Task**: `{args.task}`  ",
        f"**Split**: `{args.split}`  ",
        f"**Samples**: {len(ds)}  ",
        f"",
        f"## Metrics",
        f"",
        "| Metric | Value |",
        "|---|---|",
    ]
    for k, v in sorted(result.metrics.items()):
        md_lines.append(f"| {k} | {v:.4f} |")

    if hasattr(result, "calibration") and result.calibration:
        md_lines += ["", "## Calibration", ""]
        for k, v in result.calibration.items():
            md_lines.append(f"- **{k}**: {v:.4f}")

    (out_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    # Per-sample predictions (classification only)
    if hasattr(result, "predictions") and result.predictions:
        evaluator.save_predictions(result, output_path=out_dir / "predictions.json")

    # Confusion matrix (multiclass only)
    if hasattr(result, "confusion_matrix") and result.confusion_matrix is not None:
        import numpy as np
        _plot_confusion_matrix(
            result.confusion_matrix,
            class_names[:args.num_classes],
            output_path=out_dir / "confusion_matrix.png",
        )

    logger.info("Evaluation complete. Results → %s", out_dir)
    logger.info("Metrics: %s", result.metrics)


if __name__ == "__main__":
    main()
