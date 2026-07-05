# Inference Guide

## Overview

The `LungCarePipeline` is the primary inference API.  It runs the full
7-step pipeline end-to-end:

```
Load image → Preprocess → Classify → Generate heatmap
→ Segment → Compare to healthy → Generate report → Save outputs
```

---

## Quick Start

### Python API

```python
from inference.pipeline import LungCarePipeline, PipelineConfig
from models import create_classifier, create_segmentation_model
import torch

# Load models
classifier = create_classifier("resnet50", num_classes=6, pretrained=False)
state = torch.load("checkpoints/tb_resnet50/best.pth", map_location="cpu")
classifier.load_state_dict(state["model_state_dict"])

# Build pipeline
cfg = PipelineConfig(
    class_names=["Healthy", "Tuberculosis", "Pneumonia",
                 "COVID-19", "Lung Cancer", "Pulmonary Fibrosis"],
    task="multiclass",
    image_size=(224, 224),
    explainability_method="gradcam",
    run_segmentation=False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_version="v1.0",
)
pipeline = LungCarePipeline(classifier=classifier, config=cfg)

# Predict
result = pipeline.predict("data/patient_001.jpg", case_id="P001")

# Access structured report
print(result.report["prediction"])   # "Tuberculosis"
print(result.report["confidence"])   # 0.92
print(result.report["findings"])     # ["Opacity in upper-right lobe", ...]

# Save all outputs
pipeline.save_result(result, output_dir="results/P001", case_id="P001")
```

### CLI

```bash
python scripts/predict.py \
    --classifier checkpoints/tb_resnet50/best.pth \
    --classifier-model resnet50 \
    --input data/patient_001.jpg \
    --output results/patient_001 \
    --explainability gradcam
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `class_names` | 6 diseases | Ordered class name list |
| `task` | `multiclass` | `binary`, `multiclass`, `multilabel` |
| `image_size` | `(224, 224)` | Resize target |
| `threshold` | `0.5` | Sigmoid threshold for binary/multilabel/seg |
| `explainability_method` | `gradcam` | `gradcam`, `gradcam++`, `cam`, `rollout`, `none` |
| `run_segmentation` | `False` | Enable segmentation model |
| `amp` | `False` | AMP for GPU inference |
| `device` | `cpu` | `cpu`, `cuda`, `cuda:0` |
| `model_version` | `v1.0` | Embedded in every report |

Load from YAML:
```python
cfg = PipelineConfig.from_yaml("configs/inference_config.yaml")
```

---

## With Segmentation

```python
from models import create_segmentation_model

seg_model = create_segmentation_model("unet", in_channels=3, out_channels=1)
seg_state = torch.load("checkpoints/seg_unet/best.pth", map_location="cpu")
seg_model.load_state_dict(seg_state["model_state_dict"])

cfg = PipelineConfig(
    ...,
    run_segmentation=True,
    threshold=0.5,
)
pipeline = LungCarePipeline(
    classifier=classifier,
    config=cfg,
    segmentation_model=seg_model,
)
result = pipeline.predict("data/patient_001.jpg")

# Access mask
mask = result.segmentation["mask"]          # (H, W) uint8 array
bboxes = result.segmentation["bboxes"]     # [[x, y, w, h], ...]
coverage = result.segmentation["coverage_pct"]  # % of image covered
```

---

## With Healthy Reference Database

```python
from services.healthy_reference import HealthyReferenceDatabase, extract_features

# Build the database (one-time)
db = HealthyReferenceDatabase(dim=2048)  # ResNet50 GAP = 2048
for img_tensor in healthy_loader:
    feats = extract_features(classifier, img_tensor, device=device)
    db.add(feats, metadata={"source": "montgomery"})
db.save("data/healthy_reference.npz")

# Load and use
db = HealthyReferenceDatabase.load("data/healthy_reference.npz")
pipeline = LungCarePipeline(classifier=classifier, config=cfg, healthy_db=db)
result = pipeline.predict("data/patient_001.jpg")

# Deviation from healthy mean
deviation = result.report["scan_comparison"]["deviation_score"]
# 0 = healthy-like, ~2 = maximally different
```

---

## Batch Inference

```bash
# Process all PNGs/JPEGs in a directory
python scripts/predict.py \
    --classifier checkpoints/tb_resnet50/best.pth \
    --classifier-model resnet50 \
    --input data/test_images/ \
    --output results/batch \
    --batch

# Filter specific extensions
python scripts/predict.py \
    --input data/dicom_scans/ \
    --batch --extensions dcm
```

Output structure:
```
results/batch/
├── patient_001/
│   ├── patient_001_report.json
│   ├── patient_001_report.md
│   └── patient_001_overlay.png
├── patient_002/
│   └── ...
└── batch_summary.json    ← prediction distribution + failed files
```

---

## Report Format

Every pipeline run generates a structured JSON report:

```json
{
  "case_id": "P001",
  "status": "abnormal",
  "prediction": "Tuberculosis",
  "confidence": 0.92,
  "all_scores": {
    "Healthy": 0.03,
    "Tuberculosis": 0.92,
    "Pneumonia": 0.02,
    "COVID-19": 0.01,
    "Lung Cancer": 0.01,
    "Pulmonary Fibrosis": 0.01
  },
  "findings": [
    "Opacity detected in upper lung zone",
    "Possible cavitary lesion present",
    "Abnormal activation localised to upper-right region (score 0.87)"
  ],
  "localization": {
    "method": "GradCAM",
    "top_regions": [
      {"region": "upper-right", "score": 0.87},
      {"region": "upper-left",  "score": 0.52}
    ]
  },
  "segmentation": {
    "mask_path": "results/P001/P001_mask.png"
  },
  "scan_comparison": {
    "deviation_score": 0.62,
    "comparison_notes": [
      "Moderate deviation from healthy reference — clinical review advised"
    ]
  },
  "timestamp": "2025-01-15T09:32:00+00:00",
  "model_version": "v1.0"
}
```

---

## Explainability Selection Guide

| Method | When to use |
|---|---|
| `gradcam` | Default for any CNN; balanced speed/quality |
| `gradcam++` | When Grad-CAM misses small lesions |
| `cam` | Fastest; only for GAP+Linear heads (ResNet, DenseNet) |
| `rollout` | ViT models only; most theoretically grounded |
| `none` | When speed is critical and explanation not needed |

---

## DICOM Inference

DICOM files are loaded automatically by the pipeline:

```bash
python scripts/predict.py \
    --classifier checkpoints/tb_densenet121/best.pth \
    --classifier-model densenet121 \
    --input data/scan.dcm \
    --output results/dcm_case
```

DICOM pixel arrays are normalised to `[0, 255]` uint8 and converted to 3-channel
RGB before preprocessing. Window/level metadata is applied automatically.
