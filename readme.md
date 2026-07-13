# LungCare AI — Explainable Deep Learning for Chest X-ray Analysis

**LungCare AI** is a streamlined, production-grade PyTorch deep learning pipeline designed for automated chest X-ray analysis. It supports both classification and segmentation tasks with a focus on explainability, easy local deployment, and clinical-grade reporting.

This repository is optimized to be easy to read, easy to modify, and runnable on a standard laptop.

---

## 🌟 Key Features (Resume Highlights)

1. **PyTorch Deep Learning Pipeline:** Clean, unified `Trainer` and data loading.
2. **Chest X-ray Analysis:** Native handling for multi-disease radiography.
3. **Multi-class Disease Classification:** Full support for binary and 6-class models (Healthy, TB, Pneumonia, COVID-19, Lung Cancer, Pulmonary Fibrosis).
4. **ResNet50:** Standard robust CNN backbone.
5. **DenseNet121:** Radiologist-level feature extraction via dense blocks.
6. **EfficientNet-B0:** Compound-scaled highly efficient architecture.
7. **Vision Transformer (ViT):** Global self-attention modeling (ViT-B/16).
8. **Grad-CAM Explainability:** Includes Attention Rollout for ViT to provide interpretable heatmaps.
9. **FastAPI Deployment:** Ready-to-use REST API for inference.
10. **JSON Report Generation:** Standardized, clinical-grade reporting outputs.
11. **Mixed Precision (AMP):** Faster, memory-efficient training via `torch.amp`.
12. **TensorBoard Logging:** Built-in experiment tracking.
13. **Model Checkpointing:** Automated best-model saving and resuming.
14. **YAML Configuration:** Single master `config.yaml` controls everything.
15. **Evaluation Metrics:** AUC, per-class F1, Accuracy, Dice, and IoU via `torchmetrics`.
16. **Complete Training Pipeline:** Single `scripts/train.py` for all models.
17. **Complete Inference Pipeline:** Integrated processing via `LungCarePipeline`.

---

## 📂 Repository Structure

The repository has been refactored into a flat, modular design.

```text
lungcare_ai_ml/
├── api/                       # FastAPI Server
│   └── app.py
├── datasets/                  # Data Loading
│   ├── classification_dataset.py
│   ├── segmentation_dataset.py
│   └── transforms.py
├── evaluation/                # Reporting & Evaluation
│   ├── evaluator.py
│   └── report_generator.py
├── inference/                 # Inference Pipeline
│   └── pipeline.py
├── models/                    # Model Architectures
│   ├── __init__.py            # Lazy model factory
│   ├── densenet.py
│   ├── efficientnet.py
│   ├── gradcam.py             # Grad-CAM & Attention Rollout
│   ├── resnet.py
│   ├── unet.py                # Segmentation
│   └── vit.py
├── scripts/                   # Executables
│   └── train.py               # Main training script
├── tests/                     # Comprehensive test suite (85 tests)
├── training/                  # Core Training Logic
│   ├── losses.py              # BCE, Dice, Focal
│   ├── metrics.py
│   └── trainer.py             # Unified Trainer with AMP, Early Stopping, Checkpoints
├── config.yaml                # Single Master Configuration
├── pyproject.toml             # Dependencies & Formatting
└── requirements.txt           # Pip Requirements
```

---

## 🚀 Quick Start

### 1. Installation

Requires **Python ≥ 3.10**.

```bash
git clone https://github.com/yourname/lungcare-ai.git
cd lungcare-ai/lungcare_ai_ml
pip install -r requirements.txt
```

### 2. Configure Training

Edit `config.yaml` in the root directory. You can set the model architecture (`resnet50`, `densenet121`, `efficientnet_b0`, `vit_b16`), batch size, epochs, paths to your CSV files, and more.

### 3. Train a Model

Run the training pipeline. All settings are read from `config.yaml`.

```bash
python scripts/train.py --config config.yaml
```

The trainer will automatically:
- Load your dataset via the CSV paths specified in the config.
- Initialise the model and metrics.
- Train using Automatic Mixed Precision (AMP) if a GPU is available.
- Save TensorBoard logs to `logs/`.
- Save the best model checkpoint to `checkpoints/best.pth`.

### 4. Run the FastAPI Server

Launch the production REST API:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

- View interactive API documentation at: `http://localhost:8000/docs`
- Predict via `POST /predict` by uploading an image. The response includes the JSON report and a base64-encoded Grad-CAM heatmap.

---

## 💻 Code Examples

### Inference in Python

```python
import torch
from models import create_model
from inference.pipeline import LungCarePipeline

# 1. Load the model via factory
model = create_model("resnet50", num_classes=2, pretrained=False)

# 2. Load weights
ckpt = torch.load("checkpoints/best.pth", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 3. Initialize the pipeline
pipeline = LungCarePipeline(
    classifier=model,
    class_names=["Normal", "Tuberculosis"],
    explainability="gradcam"
)

# 4. Predict
result = pipeline.predict("path/to/xray.jpg")
print(result.report)       # JSON clinical report
print(result.heatmap)      # NumPy array Grad-CAM heatmap
```

---

## 🧪 Testing

The repository includes a comprehensive test suite (85 tests) covering datasets, models, metrics, training loops, and inference logic.

To run tests (uses CPU and synthetic data, runs in seconds):

```bash
python -m pytest tests -v
```

---

## 📜 License

This project is licensed under the MIT License.
