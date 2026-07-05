"""
Public API for the LungCare AI ``inference`` package.
"""

# Apply the torch/transformers pytree compatibility shim before any import
# that transitively loads ``transformers`` (see :mod:`utils.torch_compat`).
# Idempotent and a no-op on modern torch.
from utils.torch_compat import ensure_pytree_compat as _ensure_pytree_compat

_ensure_pytree_compat()

from inference.pipeline import LungCarePipeline, PipelineConfig, PipelineResult
from inference.postprocessing import (
    decode_classification,
    decode_segmentation,
    overlay_heatmap,
    overlay_mask,
)

__all__ = [
    "LungCarePipeline",
    "PipelineConfig",
    "PipelineResult",
    "decode_classification",
    "decode_segmentation",
    "overlay_heatmap",
    "overlay_mask",
]
