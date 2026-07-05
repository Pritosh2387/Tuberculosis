"""
Dataset download script for LungCare AI.

Downloads all supported datasets to the specified output directory.
Handles both direct HTTP downloads and Kaggle API downloads.

Supported datasets
------------------
Classification
  nih_chestxray     NIH ChestX-ray14 (Kaggle)
  montgomery        Montgomery County TB X-ray Set (NLMNIH direct)
  shenzhen          Shenzhen TB Dataset (NLMNIH direct)
  rsna              RSNA Pneumonia Detection (Kaggle)
  covidqu           COVID-QU-Ex Dataset (Kaggle)

Segmentation
  siim              SIIM-ACR Pneumothorax (Kaggle)
  mosmed            MosMedData COVID-19 CT (direct)

Usage
-----
  python scripts/download_datasets.py --dataset montgomery --output data/
  python scripts/download_datasets.py --dataset nih_chestxray --output data/
  python scripts/download_datasets.py --all --output data/

Kaggle setup
------------
  1. Go to https://www.kaggle.com/settings → Account → API → Create New Token
  2. Save the downloaded kaggle.json to:
       Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
       Linux:   ~/.kaggle/kaggle.json
  3. Ensure permissions:  chmod 600 ~/.kaggle/kaggle.json  (Linux only)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Callable

logger = logging.getLogger("lungcare.download")


# ─── Constants ────────────────────────────────────────────────────────────────

DIRECT_URLS: dict[str, dict[str, str]] = {
    "montgomery": {
        "url": "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
        "subdir": "montgomery",
        "description": "Montgomery County TB Chest X-ray Set (~54 MB)",
    },
    "shenzhen": {
        "url": "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
        "subdir": "shenzhen",
        "description": "Shenzhen TB Dataset (~300 MB)",
    },
    "mosmed": {
        "url": "https://mosmed.ai/static/files/mosmed_1110.zip",
        "subdir": "mosmed",
        "description": "MosMedData COVID-19 CT Scans (~12 GB)",
    },
}

KAGGLE_DATASETS: dict[str, dict[str, str]] = {
    "nih_chestxray": {
        "handle": "nih-chest-xrays/data",
        "subdir": "nih_chestxray",
        "description": "NIH ChestX-ray14 (~45 GB) — requires Kaggle API",
    },
    "rsna": {
        "handle": "rsna-pneumonia-detection-challenge",
        "subdir": "rsna_pneumonia",
        "competition": True,
        "description": "RSNA Pneumonia Detection (~4 GB) — requires Kaggle API",
    },
    "covidqu": {
        "handle": "anasmohammedtahir/covidqu",
        "subdir": "covidqu",
        "description": "COVID-QU-Ex Dataset (~2 GB) — requires Kaggle API",
    },
    "siim": {
        "handle": "seesee/siim-train-test",
        "subdir": "siim_pneumothorax",
        "description": "SIIM-ACR Pneumothorax (~7 GB) — requires Kaggle API",
    },
}


# ─── Download helpers ─────────────────────────────────────────────────────────


def _download_http(url: str, dest_path: Path, description: str = "") -> Path:
    """
    Download a file via HTTP with a tqdm progress bar.

    Args:
        url: Source URL.
        dest_path: Destination file path.
        description: Label shown in the progress bar.

    Returns:
        Path to the downloaded file.
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        logger.error("Install 'requests' and 'tqdm':  pip install requests tqdm")
        sys.exit(1)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        logger.info("Already exists, skipping: %s", dest_path)
        return dest_path

    logger.info("Downloading %s → %s", url, dest_path)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0))
    with (
        open(dest_path, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=description or dest_path.name) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    return dest_path


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip archive with logging."""
    logger.info("Extracting %s → %s", zip_path.name, dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Extraction complete: %s", dest_dir)


def _download_kaggle_dataset(handle: str, dest_dir: Path) -> None:
    """Download a Kaggle dataset using the kaggle API."""
    try:
        import kaggle  # type: ignore[import-untyped]
    except ImportError:
        logger.error(
            "Kaggle package not installed.  Run:  pip install kaggle\n"
            "Then set up your API token (see module docstring)."
        )
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Kaggle dataset '%s' → %s", handle, dest_dir)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(handle, path=str(dest_dir), unzip=True)
    logger.info("Kaggle dataset '%s' downloaded.", handle)


def _download_kaggle_competition(handle: str, dest_dir: Path) -> None:
    """Download a Kaggle competition dataset."""
    try:
        import kaggle  # type: ignore[import-untyped]
    except ImportError:
        logger.error("Kaggle package not installed.  Run:  pip install kaggle")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Kaggle competition '%s' → %s", handle, dest_dir)
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(handle, path=str(dest_dir), quiet=False)
    # Unzip all zip files
    for zf in dest_dir.glob("*.zip"):
        _extract_zip(zf, dest_dir)
        zf.unlink()
    logger.info("Competition '%s' downloaded.", handle)


# ─── Per-dataset downloaders ──────────────────────────────────────────────────


def download_montgomery(output_dir: Path) -> None:
    info = DIRECT_URLS["montgomery"]
    dest_dir = output_dir / info["subdir"]
    zip_path = output_dir / "montgomery.zip"
    _download_http(info["url"], zip_path, "Montgomery TB")
    _extract_zip(zip_path, dest_dir)
    zip_path.unlink(missing_ok=True)
    logger.info("Montgomery dataset ready at %s", dest_dir)


def download_shenzhen(output_dir: Path) -> None:
    info = DIRECT_URLS["shenzhen"]
    dest_dir = output_dir / info["subdir"]
    zip_path = output_dir / "shenzhen.zip"
    _download_http(info["url"], zip_path, "Shenzhen TB")
    _extract_zip(zip_path, dest_dir)
    zip_path.unlink(missing_ok=True)
    logger.info("Shenzhen dataset ready at %s", dest_dir)


def download_nih_chestxray(output_dir: Path) -> None:
    dest_dir = output_dir / KAGGLE_DATASETS["nih_chestxray"]["subdir"]
    _download_kaggle_dataset(KAGGLE_DATASETS["nih_chestxray"]["handle"], dest_dir)
    logger.info("NIH ChestX-ray14 ready at %s", dest_dir)


def download_rsna(output_dir: Path) -> None:
    dest_dir = output_dir / KAGGLE_DATASETS["rsna"]["subdir"]
    _download_kaggle_competition(
        KAGGLE_DATASETS["rsna"]["handle"], dest_dir
    )
    logger.info("RSNA Pneumonia dataset ready at %s", dest_dir)


def download_covidqu(output_dir: Path) -> None:
    dest_dir = output_dir / KAGGLE_DATASETS["covidqu"]["subdir"]
    _download_kaggle_dataset(KAGGLE_DATASETS["covidqu"]["handle"], dest_dir)
    logger.info("COVID-QU-Ex dataset ready at %s", dest_dir)


def download_siim(output_dir: Path) -> None:
    dest_dir = output_dir / KAGGLE_DATASETS["siim"]["subdir"]
    _download_kaggle_dataset(KAGGLE_DATASETS["siim"]["handle"], dest_dir)
    logger.info("SIIM-ACR Pneumothorax dataset ready at %s", dest_dir)


def download_mosmed(output_dir: Path) -> None:
    info = DIRECT_URLS["mosmed"]
    dest_dir = output_dir / info["subdir"]
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info("MosMedData appears already downloaded at %s", dest_dir)
        return
    zip_path = output_dir / "mosmed.zip"
    logger.warning(
        "MosMedData is ~12 GB.  If the direct URL is unavailable, "
        "download manually from: https://mosmed.ai/datasets/covid19_1110/"
    )
    _download_http(info["url"], zip_path, "MosMedData CT")
    _extract_zip(zip_path, dest_dir)
    zip_path.unlink(missing_ok=True)
    logger.info("MosMedData ready at %s", dest_dir)


# ─── Registry ─────────────────────────────────────────────────────────────────

_DOWNLOADERS: dict[str, Callable[[Path], None]] = {
    "montgomery": download_montgomery,
    "shenzhen": download_shenzhen,
    "nih_chestxray": download_nih_chestxray,
    "rsna": download_rsna,
    "covidqu": download_covidqu,
    "siim": download_siim,
    "mosmed": download_mosmed,
}


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LungCare AI — Dataset Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(_DOWNLOADERS.keys()),
        help="Name of the dataset to download.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download ALL supported datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Root output directory (default: ./data).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and exit.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
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

    if args.list:
        print("\nAvailable datasets:\n")
        for name, fn in _DOWNLOADERS.items():
            info = {**DIRECT_URLS, **KAGGLE_DATASETS}.get(name, {})
            desc = info.get("description", "")
            print(f"  {name:<20} {desc}")
        print()
        sys.exit(0)

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    targets: list[str] = []
    if args.all:
        targets = list(_DOWNLOADERS.keys())
    elif args.dataset:
        targets = [args.dataset]
    else:
        logger.error("Specify --dataset <name> or --all.  Use --list to see options.")
        sys.exit(1)

    for name in targets:
        logger.info("=" * 60)
        logger.info("Starting: %s", name)
        try:
            _DOWNLOADERS[name](output_dir)
            logger.info("✓ %s complete.", name)
        except Exception as exc:
            logger.error("✗ %s failed: %s", name, exc)
            if not args.all:
                sys.exit(1)

    logger.info("=" * 60)
    logger.info("All requested downloads complete.")


if __name__ == "__main__":
    main()
