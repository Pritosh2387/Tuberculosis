# Dataset Preparation Guide

## Overview

LungCare AI supports 7 datasets across classification and segmentation tasks.
This guide covers downloading, validating, and preparing each dataset.

---

## Step 1 — Download

Use the provided download script:

```bash
# See all available datasets
python scripts/download_datasets.py --list

# Download individual datasets
python scripts/download_datasets.py --dataset montgomery --output data/
python scripts/download_datasets.py --dataset shenzhen   --output data/

# Download all at once (may take several hours for large datasets)
python scripts/download_datasets.py --all --output data/
```

---

## Dataset Reference

### Montgomery County TB X-ray Set

- **Size**: ~54 MB, 138 images
- **Classes**: Normal (80), Tuberculosis (58)
- **Source**: NLMNIH (direct download, no credentials needed)
- **Includes**: Left + right lung masks for segmentation

**Expected layout after download:**
```
data/montgomery/MontgomerySet/
├── CXR_png/          ← 138 PNG images
└── ManualMask/
    ├── leftMask/     ← 138 left lung masks
    └── rightMask/    ← 138 right lung masks
```

**Naming convention:**
- `MCUCXR_XXXX_0.png` → Normal
- `MCUCXR_XXXX_1.png` → Tuberculosis

---

### Shenzhen TB Dataset

- **Size**: ~300 MB, 662 images
- **Classes**: Normal (326), Tuberculosis (336)
- **Source**: NLMNIH (direct download)

**Expected layout:**
```
data/shenzhen/ChinaSet_AllFiles/CXRs/
└── CHNCXR_XXXX_{0,1}.png
```

**Naming convention:**
- `CHNCXR_XXXX_0.png` → Normal
- `CHNCXR_XXXX_1.png` → Tuberculosis

---

### NIH ChestX-ray14

- **Size**: ~45 GB, 112,120 images
- **Classes**: 14 disease labels (multi-label); mapped to 6 unified classes
- **Source**: Kaggle — requires API token
- **Download time**: ~2–4 hours on fast internet

```bash
python scripts/download_datasets.py --dataset nih_chestxray --output data/
```

**Expected layout:**
```
data/nih_chestxray/
├── images/                    ← 112,120 PNG images
├── Data_Entry_2017.csv        ← Multi-label annotations
├── train_val_list.txt
└── test_list.txt
```

**NIH label → LungCare label mapping:**

| NIH Label | LungCare Class |
|---|---|
| No Finding | Healthy |
| Pneumonia, Infiltration | Pneumonia |
| Mass, Nodule | Lung Cancer |
| Fibrosis | Pulmonary Fibrosis |
| All others | Healthy (default) |

---

### RSNA Pneumonia Detection

- **Size**: ~4 GB, 26,684 DICOM files
- **Source**: Kaggle competition — requires API token
- **Download**: `python scripts/download_datasets.py --dataset rsna --output data/`

**Expected layout:**
```
data/rsna_pneumonia/
├── stage_2_train_images/      ← DICOM files
└── stage_2_train_labels.csv   ← patientId, x, y, width, height, Target
```

---

### COVID-QU-Ex Dataset

- **Size**: ~2 GB
- **Classes**: COVID-19, Non-COVID (viral/bacterial pneumonia), Normal
- **Source**: Kaggle — requires API token
- **Includes**: Segmentation masks (lung + infection)

**Expected layout:**
```
data/covidqu/
├── COVID-19/
│   ├── images/
│   └── lung masks/
├── Non-COVID/
│   ├── images/
│   └── lung masks/
└── Normal/
    └── images/
```

---

### SIIM-ACR Pneumothorax (Segmentation)

- **Size**: ~7 GB
- **Task**: Binary segmentation (pneumothorax vs. no pneumothorax)
- **Source**: Kaggle — requires API token

```bash
python scripts/download_datasets.py --dataset siim --output data/
```

---

### MosMedData COVID-19 CT

- **Size**: ~12 GB, 1,110 CT scans (NIfTI format)
- **Source**: mosmed.ai (direct download)
- **Note**: Large download — plan for ~1–2 hours

```bash
python scripts/download_datasets.py --dataset mosmed --output data/
```

---

## Step 2 — Prepare Data

After downloading, run the preparation script to create unified CSV manifests:

```bash
# Prepare classification data from multiple datasets
python scripts/prepare_data.py \
    --datasets montgomery shenzhen covidqu nih \
    --data-dir data/ \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42

# Output:
# data/prepared/classification/train.csv
# data/prepared/classification/val.csv
# data/prepared/classification/test.csv
```

### CSV Format (Classification)

| Column | Description |
|---|---|
| `image_path` | Absolute path to image |
| `label` | Disease class string |
| `label_idx` | Integer class index |
| `dataset` | Source dataset name |

### CSV Format (Segmentation)

| Column | Description |
|---|---|
| `image_path` | Absolute path to image |
| `mask_path` | Absolute path to binary mask PNG |
| `dataset` | Source dataset name |

---

## Step 3 — Validate Data

Check class distribution after preparation:

```python
import pandas as pd
from collections import Counter

train = pd.read_csv("data/prepared/classification/train.csv")
print("Class distribution:")
print(Counter(train["label"]))
print(f"\nTotal samples: {len(train)}")
```

---

## Handling Class Imbalance

If the dataset is highly imbalanced:

1. **Weighted loss**: Pass `--loss focal` or `--class-weights` to the training script.
2. **Oversampling**: Use `WeightedRandomSampler` (add a custom DataLoader in `train_classifier.py`).
3. **Augmentation**: Heavy augmentation on minority classes (configured in `augmentation_config.yaml`).

---

## Adding a Custom Dataset

1. Create a CSV manifest with the required columns.
2. Point the training script at your CSV:

```bash
python scripts/train_classifier.py \
    --model resnet50 \
    --train-csv data/custom/train.csv \
    --val-csv data/custom/val.csv \
    --num-classes 3 \
    --experiment custom_cls
```

3. If images are DICOM, the `DICOMDataset` class handles loading automatically.
