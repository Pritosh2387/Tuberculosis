"""inference/__init__.py — lazy re-exports to avoid torchvision chain import."""
# Import directly from inference.pipeline when needed:
#   from inference.pipeline import LungCarePipeline, PipelineResult
__all__ = ["LungCarePipeline", "PipelineResult", "decode_classification"]
