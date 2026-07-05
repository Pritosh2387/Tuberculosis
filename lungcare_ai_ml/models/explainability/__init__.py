"""
Explainability sub-package for LungCare AI.

Method selection guide
----------------------
+------------------+-------------------+-------------------------------------------+
| Method           | Architecture      | Notes                                     |
+==================+===================+===========================================+
| CAM              | GAP + Linear head | Fastest; requires specific architecture   |
| GradCAM          | Any CNN           | Works universally; slightly noisy         |
| GradCAMPlusPlus  | Any CNN           | Sharper than GradCAM; same speed          |
| AttentionRollout | ViT               | Best for ViT; accounts for all layers     |
| AttentionHeatmap | ViT               | Faster; only last-layer attention         |
+------------------+-------------------+-------------------------------------------+
"""

from models.explainability.attention_map import AttentionHeatmap, AttentionRollout
from models.explainability.cam import CAM
from models.explainability.gradcam import GradCAM, GradCAMPlusPlus

__all__ = [
    "CAM",
    "GradCAM",
    "GradCAMPlusPlus",
    "AttentionRollout",
    "AttentionHeatmap",
]
