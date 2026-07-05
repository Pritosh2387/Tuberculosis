"""
Public API for the LungCare AI ``evaluation`` package.
"""

# Apply the torch/transformers pytree compatibility shim before any import
# that transitively loads ``transformers`` (see :mod:`utils.torch_compat`).
# Idempotent and a no-op on modern torch.
from utils.torch_compat import ensure_pytree_compat as _ensure_pytree_compat

_ensure_pytree_compat()

from evaluation.evaluator import (
    ClassificationResult,
    Evaluator,
    SegmentationResult,
)
from evaluation.report_generator import ReportGenerator

__all__ = [
    "Evaluator",
    "ClassificationResult",
    "SegmentationResult",
    "ReportGenerator",
]
