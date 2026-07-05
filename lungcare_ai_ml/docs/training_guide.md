# Training Guide

## Overview

LungCare AI uses a two-stage training pipeline:

1. **Stage 1 — Classification**: Train a multi-disease classifier on chest X-rays.
2. **Stage 2 — Segmentation**: Train a segmentation model to localise lesion regions.

Both stages are fully configurable via YAML files and CLI arguments.

---

## Configuration System

Training is driven by YAML configs in `configs/`.  CLI arguments **override**
YAML values when both are provided.

### Key config sections

```yaml
# configs/classification_config.yaml (excerpt)
model:
  architecture: resnet50
  num_classes: 6
  task: multiclass            # binary | multiclass | multilabel
  pretrained: true
  dropout_rate: 0.3

training:
  epochs: 100
  batch_size: 32
  amp: true                   # Automatic mixed precision
  grad_clip: 1.0
  grad_accum_steps: 4         # Effective batch = 32 × 4 = 128
  optimizer:
    name: adamw
    lr: 1.0e-4
    weight_decay: 0.01
  scheduler:
    name: warmup_cosine
    warmup_epochs: 5
  early_stopping:
    enabled: true
    patience: 15
    mode: max
    metric: val_f1_macro
```

---

## Classification Training

### Step 1: Prepare data

```bash
python scripts/prepare_data.py \
    --datasets montgomery shenzhen covidqu nih \
    --data-dir data/ \
    --val-ratio 0.15 --test-ratio 0.15
```

Output: `data/prepared/classification/{train,val,test}.csv`

### Step 2: Train

```bash
# ResNet50 — recommended starting point
python scripts/train_classifier.py \
    --model resnet50 \
    --data-dir data/prepared/classification \
    --epochs 100 --amp --grad-accum 4 \
    --experiment cls_resnet50
```

### Step 3: Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/cls_resnet50/best.pth \
    --model resnet50 --task multiclass \
    --data-dir data/prepared/classification --split test \
    --output results/cls_resnet50
```

---

## Segmentation Training

### Step 1: Prepare segmentation data

```bash
python scripts/prepare_data.py \
    --datasets montgomery \
    --data-dir data/ \
    --val-ratio 0.15 --test-ratio 0.15
```

Output: `data/prepared/segmentation/{train,val,test}.csv`

### Step 2: Train

```bash
# Attention U-Net — best for lesion focus
python scripts/train_segmentation.py \
    --model attention_unet \
    --data-dir data/prepared/segmentation \
    --image-size 512 512 \
    --epochs 100 --amp \
    --experiment seg_attn_unet
```

---

## Training Recipes

### For Best Classification Performance

| Aspect | Setting |
|---|---|
| Architecture | `densenet121` or `efficientnet_b0` |
| Pretrained | `true` (ImageNet) |
| Scheduler | `warmup_cosine` (5 warmup epochs) |
| Loss | `label_smoothing` (ε=0.1) |
| Grad accum | 4 (effective batch ≥ 128) |
| AMP | `true` |
| Early stopping | patience=15, monitor=`val_f1_macro` |

### For Best Segmentation Performance

| Aspect | Setting |
|---|---|
| Architecture | `attention_unet` |
| Image size | 512×512 |
| Loss | `bce_dice` (50/50 weight) |
| Batch size | 4–8 (memory limited) |
| Monitor metric | `val_dice` |

---

## Gradient Accumulation

Use gradient accumulation when GPU memory limits batch size:

```bash
# Effective batch size = 8 × 4 = 32
python scripts/train_classifier.py \
    --model vit_b16 \
    --batch-size 8 --grad-accum 4
```

The training loop accumulates gradients for `grad_accum_steps` batches,
then performs one optimiser step.

---

## Automatic Mixed Precision (AMP)

AMP uses `torch.amp.autocast` to compute in `float16` where safe, and
`GradScaler` to prevent underflow.  Enable with `--amp`:

```bash
python scripts/train_classifier.py --model resnet50 --amp
```

> **Note**: AMP requires a CUDA GPU.  On CPU it is automatically disabled.

---

## Resume Training

```bash
python scripts/train_classifier.py \
    --model resnet50 \
    --resume checkpoints/cls_resnet50/checkpoint_epoch_50.pth \
    --experiment cls_resnet50
```

The checkpoint saves:
- Model weights
- Optimizer state
- Scheduler state
- AMP scaler state
- Early stopping counter
- All validation metrics

---

## Multi-GPU Training (DataParallel)

```bash
python scripts/train_classifier.py \
    --model densenet121 \
    --data-parallel \
    --experiment cls_densenet121_dp
```

> `DataParallel` wraps the model across all available GPUs automatically.
> For `DistributedDataParallel` (DDP), see the advanced section below.

---

## Monitoring with TensorBoard

```bash
# In a separate terminal:
tensorboard --logdir logs/

# Navigate to http://localhost:6006
```

Logged quantities:
- `Loss/train` and `Loss/val` per epoch
- All metric scalars (accuracy, F1, AUROC, Dice, IoU)
- Learning rate
- Per-class F1 (via `log_per_class_f1()`)
- Segmentation mask grids every N epochs

---

## Early Stopping

Early stopping is enabled by default with `patience=15` epochs.
Configure via CLI:

```bash
python scripts/train_classifier.py \
    --early-stopping-patience 20
```

Or set `early_stopping.enabled: false` in the YAML config to disable.

---

## Checkpoint Structure

All checkpoints are saved to `checkpoints/<experiment>/`.

| File | Contents |
|---|---|
| `best.pth` | Symlink / copy of the best checkpoint |
| `checkpoint_epoch_N.pth` | Full state at epoch N |

The top-3 checkpoints (by monitor metric) are kept by default.
Older checkpoints are deleted to save disk space.
