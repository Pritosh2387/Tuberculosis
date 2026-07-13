"""evaluation/__init__.py"""
from evaluation.report_generator import ReportGenerator

# Evaluator imports torchmetrics lazily; import directly when needed:
#   from evaluation.evaluator import Evaluator, ClassificationResult, SegmentationResult

__all__ = [
    "ReportGenerator",
]
