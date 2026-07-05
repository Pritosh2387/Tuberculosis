"""
Inference / prediction script for LungCare AI.

Runs the full LungCarePipeline on a single image, DICOM file, or a
directory of images and saves reports, overlays, and masks.

Usage
-----
  # Single image — classification + Grad-CAM overlay
  python scripts/predict.py \\
      --classifier checkpoints/tb_resnet50/best.pth \\
      --classifier-model resnet50 \\
      --input data/patient_001.jpg \\
      --output results/patient_001 \\
      --explainability gradcam

  # With segmentation model
  python scripts/predict.py \\
      --classifier checkpoints/tb_resnet50/best.pth \\
      --classifier-model resnet50 \\
      --segmentation checkpoints/seg_unet/best.pth \\
      --seg-model unet \\
      --input data/patient_001.jpg \\
      --output results/patient_001

  # DICOM input
  python scripts/predict.py \\
      --classifier checkpoints/tb_densenet121/best.pth \\
      --classifier-model densenet121 \\
      --input data/ct_scan.dcm \\
      --output results/ct_case

  # Batch prediction on a directory
  python scripts/predict.py \\
      --classifier checkpoints/tb_resnet50/best.pth \\
      --classifier-model resnet50 \\
      --input data/test_images/ \\
      --output results/batch \\
      --batch

  # With healthy reference database
  python scripts/predict.py \\
      --classifier checkpoints/tb_resnet50/best.pth \\
      --classifier-model resnet50 \\
      --healthy-db data/healthy_reference.npz \\
      --input data/patient_001.jpg \\
      --output results/patient_001
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("lungcare.scripts.predict")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LungCare AI — Inference / Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Models ─────────────────────────────────────────────────────────────
    parser.add_argument("--classifier", type=Path, required=True,
                        help="Path to classifier checkpoint (.pth).")
    parser.add_argument(
        "--classifier-model",
        choices=["resnet50", "densenet121", "efficientnet_b0", "vit_b16"],
        required=True,
    )
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--segmentation", type=Path, default=None,
                        help="Optional segmentation model checkpoint.")
    parser.add_argument(
        "--seg-model",
        choices=["unet", "attention_unet", "unet_plus_plus"],
        default="unet",
    )
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--healthy-db", type=Path, default=None,
                        help="Path to healthy reference .npz database.")

    # ── Input ──────────────────────────────────────────────────────────────
    parser.add_argument("--input", type=Path, required=True,
                        help="Image file or directory (with --batch).")
    parser.add_argument("--batch", action="store_true",
                        help="Process all images in --input directory.")
    parser.add_argument("--extensions", nargs="+",
                        default=["jpg", "jpeg", "png", "dcm"],
                        help="File extensions to process in batch mode.")

    # ── Pipeline settings ──────────────────────────────────────────────────
    parser.add_argument(
        "--task", choices=["binary", "multiclass", "multilabel"],
        default="multiclass",
    )
    parser.add_argument(
        "--explainability",
        choices=["gradcam", "gradcam++", "cam", "rollout", "none"],
        default="gradcam",
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")

    # ── Output ─────────────────────────────────────────────────────────────
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip saving overlay image.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1.0")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])

    return parser.parse_args()


def _load_model(
    checkpoint_path: Path,
    model_name: str,
    num_classes: int,
    in_channels: int,
    is_segmentation: bool,
) -> "torch.nn.Module":
    """Load model and restore weights from checkpoint."""
    import torch

    if is_segmentation:
        from models import create_segmentation_model
        model = create_segmentation_model(
            model_name, in_channels=in_channels, out_channels=1
        )
    else:
        from models import create_classifier
        model = create_classifier(
            model_name, num_classes=num_classes, pretrained=False
        )

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    logger.info(
        "Loaded %s checkpoint (epoch %s) from %s",
        model_name, state.get("epoch", "?"), checkpoint_path.name,
    )
    return model


def _predict_one(
    pipeline: "LungCarePipeline",
    image_path: Path,
    output_dir: Path,
    save_overlay: bool,
) -> dict:
    """Run pipeline on one image and save results."""
    case_id = image_path.stem
    result = pipeline.predict(image_path, case_id=case_id)

    saved = pipeline.save_result(
        result, output_dir=output_dir, case_id=case_id
    )

    if not save_overlay and "overlay" in saved:
        saved["overlay"].unlink(missing_ok=True)
        del saved["overlay"]

    logger.info(
        "[%s] → %s (conf=%.2f, time=%.2fs)",
        case_id,
        result.report["prediction"],
        result.report["confidence"],
        result.inference_time_s,
    )
    return result.report


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load classifier ────────────────────────────────────────────────────────
    classifier = _load_model(
        checkpoint_path=args.classifier.resolve(),
        model_name=args.classifier_model,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        is_segmentation=False,
    )

    # ── Load segmentation model (optional) ────────────────────────────────────
    seg_model = None
    if args.segmentation and args.segmentation.exists():
        seg_model = _load_model(
            checkpoint_path=args.segmentation.resolve(),
            model_name=args.seg_model,
            num_classes=1,
            in_channels=args.in_channels,
            is_segmentation=True,
        )

    # ── Load healthy reference DB (optional) ──────────────────────────────────
    healthy_db = None
    if args.healthy_db and args.healthy_db.exists():
        from services.healthy_reference import HealthyReferenceDatabase
        healthy_db = HealthyReferenceDatabase.load(args.healthy_db)
        logger.info("Healthy DB loaded: %d reference vectors.", len(healthy_db))

    # ── Build pipeline ────────────────────────────────────────────────────────
    from inference.pipeline import LungCarePipeline, PipelineConfig

    class_names: list[str] = [
        "Healthy", "Tuberculosis", "Pneumonia",
        "COVID-19", "Lung Cancer", "Pulmonary Fibrosis",
    ]

    cfg = PipelineConfig(
        class_names=class_names[:args.num_classes],
        task=args.task,
        image_size=tuple(args.image_size),
        threshold=args.threshold,
        explainability_method=args.explainability,
        run_segmentation=seg_model is not None,
        amp=args.amp,
        device=device,
        model_version=args.model_version,
    )
    pipeline = LungCarePipeline(
        classifier=classifier,
        config=cfg,
        segmentation_model=seg_model,
        healthy_db=healthy_db,
    )

    # ── Collect input files ───────────────────────────────────────────────────
    input_path = args.input.resolve()
    if args.batch:
        if not input_path.is_dir():
            logger.error("--batch requires --input to be a directory.")
            sys.exit(1)
        exts = {f".{e.lstrip('.')}" for e in args.extensions}
        image_files = [
            p for p in sorted(input_path.rglob("*"))
            if p.is_file() and p.suffix.lower() in exts
        ]
        logger.info("Batch mode: %d files found in %s", len(image_files), input_path)
    else:
        if not input_path.exists():
            logger.error("Input not found: %s", input_path)
            sys.exit(1)
        image_files = [input_path]

    # ── Run inference ─────────────────────────────────────────────────────────
    output_dir = args.output.resolve()
    all_reports: list[dict] = []
    failed: list[str] = []

    for img_path in image_files:
        try:
            # In batch mode, create per-image subdirectory
            if args.batch:
                case_out = output_dir / img_path.stem
            else:
                case_out = output_dir

            report = _predict_one(
                pipeline=pipeline,
                image_path=img_path,
                output_dir=case_out,
                save_overlay=not args.no_overlay,
            )
            all_reports.append(report)
        except Exception as exc:
            logger.error("Failed on %s: %s", img_path.name, exc)
            failed.append(str(img_path))

    # ── Batch summary ─────────────────────────────────────────────────────────
    if args.batch and all_reports:
        summary_path = output_dir / "batch_summary.json"
        summary = {
            "total": len(image_files),
            "succeeded": len(all_reports),
            "failed": len(failed),
            "failed_files": failed,
            "predictions": [
                {
                    "case_id": r["case_id"],
                    "prediction": r["prediction"],
                    "confidence": r["confidence"],
                    "status": r["status"],
                }
                for r in all_reports
            ],
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Batch summary saved → %s", summary_path)

        # Print quick distribution
        from collections import Counter
        dist = Counter(r["prediction"] for r in all_reports)
        logger.info("Prediction distribution: %s", dict(dist))

    if failed:
        logger.warning("%d file(s) failed to process.", len(failed))

    logger.info("Inference complete. Results → %s", output_dir)


if __name__ == "__main__":
    main()
