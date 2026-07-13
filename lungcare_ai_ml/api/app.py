"""
api/app.py
───────────
FastAPI REST API for LungCare AI inference.

Endpoints
---------
GET  /health          — health check + model info
POST /predict         — classify one chest X-ray image
GET  /classes         — list supported disease classes
GET  /config          — return current inference configuration

Usage
-----
    # Start the server
    uvicorn api.app:app --host 0.0.0.0 --port 8000

    # Predict via curl
    curl -X POST http://localhost:8000/predict \\
         -F "file=@data/patient_001.png" \\
         -F "case_id=patient_001"

Interview notes
---------------
Why FastAPI over Flask?
  FastAPI uses Python type hints to auto-generate OpenAPI docs at /docs.
  It uses async I/O (Starlette) for high-throughput endpoints.
  Pydantic response models ensure the JSON schema is always enforced.

Why UploadFile instead of a URL parameter?
  Medical images live behind firewalls — callers upload the file
  directly rather than providing a publicly accessible URL.
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# ── Add project root to sys.path (allows running as: uvicorn api.app:app) ────
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference.pipeline import LungCarePipeline
from models import create_model
from utils.config import load_config

logger = logging.getLogger("lungcare.api")

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LungCare AI",
    description=(
        "Explainable Deep Learning for Chest X-ray Analysis.\n"
        "Supports ResNet50, DenseNet121, EfficientNet-B0, ViT.\n"
        "Returns disease classification + Grad-CAM heatmap + JSON report."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ─────────────────────────────────────────────────────────────

_pipeline: LungCarePipeline | None = None
_config:   Any = None


def _get_pipeline() -> LungCarePipeline:
    """Lazy-load the inference pipeline on first request."""
    global _pipeline, _config

    if _pipeline is not None:
        return _pipeline

    cfg = load_config("config.yaml")
    _config = cfg
    ic  = cfg.inference

    # Build model
    model = create_model(
        cfg.model.architecture,
        num_classes=cfg.data.num_classes,
        pretrained=False,
        dropout_rate=cfg.model.dropout_rate,
    )

    # Load checkpoint if available
    ckpt_path = Path(ic.checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=ic.device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state", ckpt))
        logger.info("Loaded checkpoint: %s", ckpt_path)
    else:
        logger.warning(
            "Checkpoint not found at %s — using random weights. "
            "Run training first.", ckpt_path
        )

    _pipeline = LungCarePipeline(
        classifier=model,
        class_names=cfg.data.class_names,
        image_size=cfg.data.image_size,
        device=ic.device,
        explainability=ic.explainability,
        run_segmentation=ic.run_segmentation,
        threshold=ic.threshold,
        model_version=cfg.project.version,
    )
    return _pipeline


# ─── Response models ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model:         str
    num_classes:   int
    device:        str
    checkpoint:    str


class PredictResponse(BaseModel):
    case_id:          str | None
    prediction:       str
    confidence:       float
    all_scores:       dict[str, float]
    status:           str           # "normal" | "abnormal"
    findings:         list[str]
    localization:     dict[str, Any]
    gradcam_image_b64: str | None   # base64-encoded PNG of the overlay
    inference_time_s: float
    model_version:    str


class ClassesResponse(BaseModel):
    classes:     list[str]
    num_classes: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["status"])
def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Returns the model name, number of classes, device, and checkpoint path.
    Use this to verify the server is running and the model is loaded.
    """
    cfg = load_config("config.yaml")
    ic  = cfg.inference
    return HealthResponse(
        status="ok",
        model=cfg.model.architecture,
        num_classes=cfg.data.num_classes,
        device=ic.device,
        checkpoint=ic.checkpoint_path,
    )


@app.get("/classes", response_model=ClassesResponse, tags=["info"])
def list_classes() -> ClassesResponse:
    """
    Return the list of disease classes the model was trained on.

    Architecture supports 6 classes; default deployment uses 2 (TB binary).
    """
    cfg = load_config("config.yaml")
    return ClassesResponse(
        classes=cfg.data.class_names,
        num_classes=cfg.data.num_classes,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG or PNG)"),
    case_id: str = Form(default="", description="Optional patient/case identifier"),
) -> PredictResponse:
    """
    Classify a chest X-ray image.

    Upload a JPEG or PNG chest X-ray. Returns:
    - Predicted disease class + confidence
    - Probability scores for all classes
    - Clinical findings list
    - Grad-CAM activation regions
    - Base64-encoded Grad-CAM overlay image
    - Structured JSON report (embedded in response)
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. "
                   "Upload a JPEG or PNG image.",
        )

    # Read and decode image
    try:
        contents  = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np  = np.array(pil_image, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Image decode failed: {exc}")

    # Run pipeline
    try:
        pipeline = _get_pipeline()
        result   = pipeline.predict(
            image_input=image_np,
            case_id=case_id or None,
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    return PredictResponse(
        case_id=result.report["case_id"],
        prediction=result.report["prediction"],
        confidence=result.report["confidence"],
        all_scores=result.report["all_scores"],
        status=result.report["status"],
        findings=result.report["findings"],
        localization=result.report["localization"],
        gradcam_image_b64=result.gradcam_as_b64_png(),
        inference_time_s=result.inference_time_s,
        model_version=result.report["model_version"],
    )


@app.get("/config", tags=["info"])
def get_config() -> dict[str, Any]:
    """Return the active inference configuration (from config.yaml)."""
    cfg = load_config("config.yaml")
    ic  = cfg.inference
    return {
        "model":           cfg.model.architecture,
        "num_classes":     cfg.data.num_classes,
        "class_names":     cfg.data.class_names,
        "image_size":      cfg.data.image_size,
        "device":          ic.device,
        "explainability":  ic.explainability,
        "threshold":       ic.threshold,
        "run_segmentation": ic.run_segmentation,
    }
