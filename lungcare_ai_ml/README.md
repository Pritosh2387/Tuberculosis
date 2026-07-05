# LungCare AI — Explainable Deep Learning for Chest X-ray Analysis

<p align="center">
  <img src="docs/assets/banner_placeholder.png" alt="LungCare AI Banner" width="100%"/>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3120/"><img src="https://img.shields.io/badge/Python-3.12-blue.svg" /></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-orange.svg" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" /></a>
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg" />
  <img src="https://img.shields.io/badge/CUDA-Supported-76B900.svg" />
</p>

---

## Overview

**LungCare AI** is a production-grade explainable deep learning system for automated chest X-ray and CT scan analysis. It performs multi-disease classification, lesion localisation, and segmentation — with clinical-grade structured report generation.

### What it does

| Task | Output |
|---|---|
| **Normal vs. Abnormal** | Binary classification with confidence score |
| **Multi-disease classification** | 6-class: Healthy, TB, Pneumonia, COVID-19, Lung Cancer, Pulmonary Fibrosis |
| **Lesion localisation** | Grad-CAM / Attention Rollout heatmaps |
| **Lesion segmentation** | Binary mask via U-Net / Attention U-Net / U-Net++ |
| **Healthy comparison** | Deviation score from healthy reference database |
| **Structured report** | JSON + Markdown clinical report with findings |

### Example output

```json
{
  "status": "abnormal",
  "prediction": "Tuberculosis",
  "confidence": 0.92,
  "findings": [
    "Opacity detected in upper lung zone",
    "Possible cavitary lesion present",
    "Abnormal activation localised to upper-right region (score 0.87)"
  ],
  "localization": {
    "method": "GradCAM",
    "top_regions": [{"region": "upper-right", "score": 0.87}]
  }
}
```

---

## Repository Structure

```
lungcare_ai_ml/
├── configs/                   # YAML configuration files
│   ├── base_config.yaml
│   ├── classification_config.yaml
│   ├── segmentation_config.yaml
│   ├── augmentation_config.yaml
│   └── inference_config.yaml
│
├── datasets/                  # PyTorch Dataset classes
│   ├── base_dataset.py
│   ├── classification_dataset.py
│   ├── segmentation_dataset.py
│   ├── dicom_dataset.py
│   └── transforms.py
│
├── models/                    # All model architectures
│   ├── classification/        # ResNet50, DenseNet121, EfficientNet-B0, ViT-B/16
│   ├── segmentation/          # U-Net, Attention U-Net, U-Net++
│   └── explainability/        # CAM, Grad-CAM, Grad-CAM++, Attention Rollout
│
├── training/                  # Training loop, losses, metrics, schedulers
│   ├── trainer.py             # Abstract BaseTrainer
│   ├── classification_trainer.py
│   ├── segmentation_trainer.py
│   ├── losses.py
│   ├── metrics.py
│   └── schedulers.py
│
├── evaluation/                # Evaluator + report generator
├── inference/                 # Production inference pipeline
├── services/                  # Healthy reference database
├── utils/                     # Logging, seeding, config, checkpoints, DICOM
│
├── scripts/                   # CLI entry points
│   ├── download_datasets.py
│   ├── prepare_data.py
│   ├── train_classifier.py
│   ├── train_segmentation.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/                     # pytest test suite
├── docs/                      # Extended documentation
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourname/lungcare-ai.git
cd lungcare-ai/lungcare_ai_ml
pip install -r requirements.txt
```

See [docs/installation.md](docs/installation.md) for GPU, DICOM, and virtual environment setup.

### 2. Download Datasets

```bash
# Free direct-download datasets (no credentials needed)
python scripts/download_datasets.py --dataset montgomery --output data/
python scripts/download_datasets.py --dataset shenzhen   --output data/

# Kaggle datasets (requires API token)
python scripts/download_datasets.py --dataset nih_chestxray --output data/
python scripts/download_datasets.py --dataset covidqu       --output data/
```

### 3. Prepare Data

```bash
python scripts/prepare_data.py \
    --datasets montgomery shenzhen covidqu \
    --data-dir data/ \
    --val-ratio 0.15 --test-ratio 0.15
```

### 4. Train a Classifier

```bash
python scripts/train_classifier.py \
    --model resnet50 \
    --data-dir data/prepared/classification \
    --epochs 100 --amp --grad-accum 4 \
    --experiment tb_resnet50
```

### 5. Run Inference

```bash
python scripts/predict.py \
    --classifier checkpoints/tb_resnet50/best.pth \
    --classifier-model resnet50 \
    --input data/patient_001.jpg \
    --output results/patient_001 \
    --explainability gradcam
```

---

## Model Zoo

| Architecture | Task | Backbone | Params | Notes |
|---|---|---|---|---|
| `ResNet50Classifier` | Classification | ResNet-50 | ~23M | Grad-CAM via `layer4` |
| `DenseNet121Classifier` | Classification | DenseNet-121 | ~7M | Grad-CAM via `denseblock4` |
| `EfficientNetB0Classifier` | Classification | EfficientNet-B0 (timm) | ~5M | Grad-CAM via `conv_head` |
| `ViTClassifier` | Classification | ViT-Base/16 (timm) | ~86M | Attention Rollout |
| `UNet` | Segmentation | Custom encoder-decoder | ~7–31M | Variable features |
| `AttentionUNet` | Segmentation | U-Net + Attention Gates | ~9–35M | `get_attention_maps()` |
| `UNetPlusPlus` | Segmentation | Dense nested skip | ~10–50M | Deep supervision |

---

## Supported Datasets

| Dataset | Disease | Type | Source |
|---|---|---|---|
| NIH ChestX-ray14 | Multi-disease | CXR PNG | Kaggle |
| Montgomery County | Tuberculosis | CXR PNG | NLMNIH direct |
| Shenzhen | Tuberculosis | CXR PNG | NLMNIH direct |
| RSNA Pneumonia | Pneumonia | CXR DICOM | Kaggle |
| COVID-QU-Ex | COVID-19 | CXR PNG | Kaggle |
| SIIM-ACR Pneumothorax | Segmentation masks | CXR + mask | Kaggle |
| MosMedData | COVID-19 CT | CT NIfTI | Direct |

---

## Explainability Methods

| Method | Class | Architectures | Notes |
|---|---|---|---|
| CAM | `CAM` | GAP + Linear head | Fastest |
| Grad-CAM | `GradCAM` | Any CNN | Universal |
| Grad-CAM++ | `GradCAMPlusPlus` | Any CNN | Sharper than Grad-CAM |
| Attention Rollout | `AttentionRollout` | ViT | Layer-wise rollout |
| Attention Heatmap | `AttentionHeatmap` | ViT | Last-layer only, faster |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Fast unit tests only (skip integration)
pytest tests/ -v -k "not integration"

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---
