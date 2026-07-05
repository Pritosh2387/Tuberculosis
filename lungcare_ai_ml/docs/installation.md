# Installation Guide

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.12 | 3.12 |
| PyTorch | 2.0 | 2.2 |
| CUDA | 11.8 | 12.1 |
| RAM | 8 GB | 32 GB |
| GPU VRAM | — | 8 GB (16 GB for ViT) |
| Disk | 5 GB | 200 GB (with NIH dataset) |

---

## Windows Setup

```powershell
# 1. Clone the repository
git clone https://github.com/yourname/lungcare-ai.git
cd lungcare-ai\lungcare_ai_ml

# 2. Create a virtual environment (Python 3.12)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (CUDA 12.1 — adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install all other dependencies
pip install -r requirements.txt
```

> **Note**: If you don't have a GPU, install the CPU-only PyTorch:
> ```powershell
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### PowerShell Execution Policy

If you get an error running `.venv\Scripts\Activate.ps1`:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Linux / macOS Setup

```bash
# 1. Clone
git clone https://github.com/yourname/lungcare-ai.git
cd lungcare-ai/lungcare_ai_ml

# 2. Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Dependencies
pip install -r requirements.txt
```

---

## DICOM Support

DICOM files (`.dcm`) are supported via `pydicom`.  It is included in
`requirements.txt`.  No additional setup is needed for standard DICOM files.

For compressed DICOM (JPEG 2000 / RLE), install the optional codec:
```bash
pip install pylibjpeg pylibjpeg-libjpeg gdcm
```

---

## Kaggle API Setup

Required for downloading NIH ChestX-ray14, RSNA Pneumonia, COVID-QU-Ex,
and SIIM-ACR Pneumothorax datasets.

1. Create an account at [kaggle.com](https://www.kaggle.com)
2. Go to **Settings → Account → API → Create New Token**
3. A `kaggle.json` file will be downloaded
4. Place it in the correct location:

**Windows:**
```
C:\Users\<YourUsername>\.kaggle\kaggle.json
```

**Linux / macOS:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

5. Install the Kaggle package:
```bash
pip install kaggle
```

6. Test the setup:
```bash
kaggle datasets list
```

---

## Verify Installation

```bash
python -c "
import torch, timm, albumentations, torchmetrics, pydantic
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
print('timm:', timm.__version__)
print('All imports OK')
"
```

Expected output (GPU system):
```
PyTorch: 2.2.0+cu121
CUDA available: True
CUDA device: NVIDIA GeForce RTX 3080
timm: 0.9.x
All imports OK
```

---

## Running the Test Suite

```bash
# All tests (requires ~5 min on CPU)
pytest tests/ -v

# Quick smoke test (unit tests only, ~30s)
pytest tests/ -v -k "not integration"

# With coverage report
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html  # Linux/macOS
start htmlcov/index.html # Windows
```
