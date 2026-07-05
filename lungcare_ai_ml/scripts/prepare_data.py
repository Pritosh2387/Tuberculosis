"""
Data preparation script for LungCare AI.

Organises downloaded raw datasets into a unified folder structure,
generates train/val/test CSV manifests, and validates data integrity.

Supported dataset processors
-----------------------------
  montgomery    Montgomery County TB X-ray Set
  shenzhen      Shenzhen TB Dataset
  nih           NIH ChestX-ray14
  rsna          RSNA Pneumonia Detection
  covidqu       COVID-QU-Ex
  siim          SIIM-ACR Pneumothorax (segmentation)

Expected input layout after download_datasets.py
-------------------------------------------------
  data/
  ├── montgomery/
  │   └── MontgomerySet/
  │       ├── CXR_png/           ← X-ray images
  │       └── ManualMask/        ← Lung masks (left + right)
  ├── shenzhen/
  │   └── ChinaSet_AllFiles/
  │       └── CXRs/              ← X-ray images
  ├── nih_chestxray/
  │   ├── images/                ← 112,120 PNG images
  │   └── Data_Entry_2017.csv   ← Labels file
  ├── rsna_pneumonia/
  │   ├── stage_2_train_images/
  │   └── stage_2_train_labels.csv
  └── covidqu/
      ├── COVID-19/
      ├── Non-COVID/
      └── Normal/

Output layout
-------------
  data/prepared/
  ├── classification/
  │   ├── train.csv
  │   ├── val.csv
  │   └── test.csv
  └── segmentation/
      ├── train.csv
      ├── val.csv
      └── test.csv

CSV columns (classification):  image_path, label, label_idx, dataset
CSV columns (segmentation):    image_path, mask_path, dataset

Usage
-----
  python scripts/prepare_data.py --datasets montgomery shenzhen nih --data-dir data/
  python scripts/prepare_data.py --all --data-dir data/ --val-ratio 0.15 --test-ratio 0.15
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("lungcare.prepare")

# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES: list[str] = [
    "Healthy",
    "Tuberculosis",
    "Pneumonia",
    "COVID-19",
    "Lung Cancer",
    "Pulmonary Fibrosis",
]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}

# NIH disease label → unified class mapping
_NIH_MAP: dict[str, str] = {
    "No Finding": "Healthy",
    "Pneumonia": "Pneumonia",
    "Mass": "Lung Cancer",
    "Nodule": "Lung Cancer",
    "Fibrosis": "Pulmonary Fibrosis",
    "Infiltration": "Pneumonia",
    "Consolidation": "Pneumonia",
    "Effusion": "Pneumonia",
}


# ─── Record type ─────────────────────────────────────────────────────────────

Record = dict[str, str]


# ─── Dataset processors ───────────────────────────────────────────────────────


def process_montgomery(data_dir: Path) -> tuple[list[Record], list[Record]]:
    """
    Process Montgomery County TB dataset.

    Returns:
        (classification_records, segmentation_records)
    """
    base = data_dir / "montgomery" / "MontgomerySet"
    img_dir = base / "CXR_png"
    mask_l = base / "ManualMask" / "leftMask"
    mask_r = base / "ManualMask" / "rightMask"

    if not img_dir.exists():
        logger.warning("Montgomery images not found at %s", img_dir)
        return [], []

    cls_records: list[Record] = []
    seg_records: list[Record] = []

    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        # Naming convention: MCUCXR_XXXX_0 = Normal, MCUCXR_XXXX_1 = TB
        label = "Tuberculosis" if name.endswith("_1") else "Healthy"
        cls_records.append(
            {
                "image_path": str(img_path),
                "label": label,
                "label_idx": str(CLASS_TO_IDX[label]),
                "dataset": "montgomery",
            }
        )
        # Masks: left + right lung masks
        ml = mask_l / img_path.name
        mr = mask_r / img_path.name
        if ml.exists() and mr.exists():
            seg_records.append(
                {
                    "image_path": str(img_path),
                    "mask_left_path": str(ml),
                    "mask_right_path": str(mr),
                    "dataset": "montgomery",
                }
            )

    logger.info("Montgomery: %d classification, %d segmentation", len(cls_records), len(seg_records))
    return cls_records, seg_records


def process_shenzhen(data_dir: Path) -> list[Record]:
    """
    Process Shenzhen TB dataset.

    Image naming convention: CHNCXR_XXXX_0 = Normal, CHNCXR_XXXX_1 = TB
    """
    img_dir = data_dir / "shenzhen" / "ChinaSet_AllFiles" / "CXRs"
    if not img_dir.exists():
        logger.warning("Shenzhen images not found at %s", img_dir)
        return []

    records: list[Record] = []
    for img_path in sorted(img_dir.glob("*.png")):
        name = img_path.stem
        label = "Tuberculosis" if name.endswith("_1") else "Healthy"
        records.append(
            {
                "image_path": str(img_path),
                "label": label,
                "label_idx": str(CLASS_TO_IDX[label]),
                "dataset": "shenzhen",
            }
        )

    logger.info("Shenzhen: %d records", len(records))
    return records


def process_nih(data_dir: Path) -> list[Record]:
    """
    Process NIH ChestX-ray14 dataset.

    Uses the multi-label CSV; maps NIH disease labels to unified classes.
    """
    img_dir = data_dir / "nih_chestxray" / "images"
    csv_path = data_dir / "nih_chestxray" / "Data_Entry_2017.csv"

    if not csv_path.exists():
        logger.warning("NIH CSV not found at %s", csv_path)
        return []

    records: list[Record] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_file = img_dir / row["Image Index"]
            if not img_file.exists():
                continue

            nih_labels = row["Finding Labels"].split("|")
            # Map to unified label; take first recognisable mapping
            unified = "Healthy"
            for nih_lbl in nih_labels:
                if nih_lbl.strip() in _NIH_MAP:
                    unified = _NIH_MAP[nih_lbl.strip()]
                    if unified != "Healthy":
                        break

            records.append(
                {
                    "image_path": str(img_file),
                    "label": unified,
                    "label_idx": str(CLASS_TO_IDX[unified]),
                    "dataset": "nih",
                }
            )

    logger.info("NIH ChestX-ray14: %d records", len(records))
    return records


def process_rsna(data_dir: Path) -> list[Record]:
    """Process RSNA Pneumonia Detection dataset."""
    img_dir = data_dir / "rsna_pneumonia" / "stage_2_train_images"
    csv_path = data_dir / "rsna_pneumonia" / "stage_2_train_labels.csv"

    if not csv_path.exists():
        logger.warning("RSNA CSV not found at %s", csv_path)
        return []

    # RSNA CSV: patientId, x, y, width, height, Target (1=pneumonia, 0=normal)
    seen: set[str] = set()
    records: list[Record] = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["patientId"]
            if pid in seen:
                continue
            seen.add(pid)

            # RSNA images are DICOM; check for .dcm files
            dcm = img_dir / f"{pid}.dcm"
            label = "Pneumonia" if row.get("Target", "0") == "1" else "Healthy"
            if dcm.exists():
                records.append(
                    {
                        "image_path": str(dcm),
                        "label": label,
                        "label_idx": str(CLASS_TO_IDX[label]),
                        "dataset": "rsna",
                    }
                )

    logger.info("RSNA: %d records", len(records))
    return records


def process_covidqu(data_dir: Path) -> list[Record]:
    """
    Process COVID-QU-Ex dataset.

    Expected folder structure: COVID-19/, Non-COVID/, Normal/
    """
    base = data_dir / "covidqu"
    folder_label_map = {
        "COVID-19": "COVID-19",
        "Non-COVID": "Pneumonia",
        "Normal": "Healthy",
    }
    records: list[Record] = []
    for folder, label in folder_label_map.items():
        img_dir = base / folder
        if not img_dir.exists():
            logger.warning("COVID-QU-Ex folder not found: %s", img_dir)
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_path in sorted(img_dir.rglob(ext)):
                records.append(
                    {
                        "image_path": str(img_path),
                        "label": label,
                        "label_idx": str(CLASS_TO_IDX[label]),
                        "dataset": "covidqu",
                    }
                )
    logger.info("COVID-QU-Ex: %d records", len(records))
    return records


# ─── Split & write ────────────────────────────────────────────────────────────


def stratified_split(
    records: list[Record],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Record], list[Record], list[Record]]:
    """
    Stratified train/val/test split by 'label' field.

    Returns:
        (train_records, val_records, test_records)
    """
    rng = random.Random(seed)

    # Group by label
    groups: dict[str, list[Record]] = {}
    for rec in records:
        lbl = rec.get("label", "unknown")
        groups.setdefault(lbl, []).append(rec)

    train, val, test = [], [], []
    for lbl, recs in groups.items():
        rng.shuffle(recs)
        n = len(recs)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        test.extend(recs[:n_test])
        val.extend(recs[n_test : n_test + n_val])
        train.extend(recs[n_test + n_val :])

    return train, val, test


def write_csv(records: list[Record], path: Path) -> None:
    """Write records to a CSV file."""
    if not records:
        logger.warning("No records to write to %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info("Wrote %d records → %s", len(records), path)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LungCare AI — Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["montgomery", "shenzhen", "nih", "rsna", "covidqu"],
        help="Datasets to process.",
    )
    parser.add_argument("--all", action="store_true", help="Process all datasets.")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Root data directory (default: ./data).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for prepared CSVs (default: <data-dir>/prepared).",
    )
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = args.data_dir.resolve()
    out_dir = (args.output_dir or data_dir / "prepared").resolve()

    datasets = (
        ["montgomery", "shenzhen", "nih", "rsna", "covidqu"]
        if args.all
        else (args.datasets or [])
    )
    if not datasets:
        logger.error("Specify --datasets or --all.")
        sys.exit(1)

    all_cls: list[Record] = []
    all_seg: list[Record] = []

    for ds in datasets:
        logger.info("Processing: %s", ds)
        if ds == "montgomery":
            cls_recs, seg_recs = process_montgomery(data_dir)
            all_cls.extend(cls_recs)
            all_seg.extend(seg_recs)
        elif ds == "shenzhen":
            all_cls.extend(process_shenzhen(data_dir))
        elif ds == "nih":
            all_cls.extend(process_nih(data_dir))
        elif ds == "rsna":
            all_cls.extend(process_rsna(data_dir))
        elif ds == "covidqu":
            all_cls.extend(process_covidqu(data_dir))

    # ── Classification splits ─────────────────────────────────────────────────
    if all_cls:
        train, val, test = stratified_split(
            all_cls, args.val_ratio, args.test_ratio, args.seed
        )
        cls_dir = out_dir / "classification"
        write_csv(train, cls_dir / "train.csv")
        write_csv(val,   cls_dir / "val.csv")
        write_csv(test,  cls_dir / "test.csv")
        logger.info(
            "Classification split: train=%d | val=%d | test=%d",
            len(train), len(val), len(test),
        )
        # Class distribution
        from collections import Counter
        dist = Counter(r["label"] for r in all_cls)
        logger.info("Class distribution: %s", dict(dist))

    # ── Segmentation splits ───────────────────────────────────────────────────
    if all_seg:
        seg_train, seg_val, seg_test = stratified_split(
            all_seg, args.val_ratio, args.test_ratio, args.seed
        )
        seg_dir = out_dir / "segmentation"
        write_csv(seg_train, seg_dir / "train.csv")
        write_csv(seg_val,   seg_dir / "val.csv")
        write_csv(seg_test,  seg_dir / "test.csv")
        logger.info(
            "Segmentation split: train=%d | val=%d | test=%d",
            len(seg_train), len(seg_val), len(seg_test),
        )

    logger.info("Data preparation complete. Output: %s", out_dir)


if __name__ == "__main__":
    main()
