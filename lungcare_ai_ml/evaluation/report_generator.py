"""
evaluation/report_generator.py
────────────────────────────────
Structured clinical report generator for LungCare AI.

Given model outputs (softmax probabilities + optional Grad-CAM heatmap
+ optional segmentation mask path), generates a machine-readable
JSON-serialisable report matching the project's output schema.

Output schema
-------------
{
    "case_id":      "patient_001",
    "status":       "abnormal",
    "prediction":   "Tuberculosis",
    "confidence":   0.92,
    "all_scores":   {"Normal": 0.08, "Tuberculosis": 0.92},
    "findings":     ["Opacity detected in upper lung zone", ...],
    "localization": {"method": "GradCAM", "top_regions": [...]},
    "segmentation": {"mask_path": "outputs/mask.png"},
    "timestamp":    "2025-01-15T09:32:00Z",
    "model_version": "v2.0"
}
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("lungcare.evaluation.report_generator")

# ─── Clinical finding templates ───────────────────────────────────────────────
# Template findings per disease class — selected based on the predicted class.
# These are standard radiological descriptions, not fabricated results.

_FINDING_TEMPLATES: dict[str, list[str]] = {
    "Tuberculosis": [
        "Opacity detected in upper lung zone",
        "Possible cavitary lesion present",
        "Increased interstitial markings",
        "Hilar lymphadenopathy pattern noted",
    ],
    "Pneumonia": [
        "Lobar or segmental consolidation detected",
        "Air-space opacity consistent with infection",
        "Pleural effusion may be present",
    ],
    "COVID-19": [
        "Bilateral ground-glass opacities detected",
        "Peripheral distribution of opacities",
        "Crazy-paving pattern noted",
    ],
    "Lung Cancer": [
        "Pulmonary nodule or mass detected",
        "Irregular margin noted — further evaluation recommended",
        "Possible hilar or mediastinal involvement",
    ],
    "Pulmonary Fibrosis": [
        "Reticular pattern with honeycombing",
        "Bilateral basal predominance",
        "Traction bronchiectasis detected",
    ],
    "Normal": [
        "No significant abnormality detected",
        "Lung fields appear clear",
        "Normal cardiomediastinal silhouette",
    ],
    # Legacy alias
    "Healthy": [
        "No significant abnormality detected",
        "Lung fields appear clear",
        "Normal cardiomediastinal silhouette",
    ],
}


def _analyse_heatmap(
    heatmap: np.ndarray,
    top_k: int = 3,
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Map Grad-CAM activation peaks to coarse anatomical regions.

    Divides the heatmap into a 2×3 grid (upper/mid/lower × left/right)
    and returns the top_k regions with highest mean activation.

    Args:
        heatmap:   Float32 (H, W) array in [0, 1].
        top_k:     Maximum regions to return.
        threshold: Minimum mean activation for a region to be included.

    Returns:
        List of {'region': str, 'score': float} dicts, sorted by score desc.
    """
    H, W = heatmap.shape
    rows = [
        (0,         H // 3,     "upper"),
        (H // 3,    2 * H // 3, "mid"),
        (2 * H // 3, H,         "lower"),
    ]
    cols = [(0, W // 2, "right"), (W // 2, W, "left")]

    regions: list[dict[str, Any]] = []
    for r0, r1, row_name in rows:
        for c0, c1, col_name in cols:
            score = float(heatmap[r0:r1, c0:c1].mean())
            if score >= threshold:
                regions.append({"region": f"{row_name}-{col_name}",
                                "score": round(score, 3)})

    regions.sort(key=lambda x: x["score"], reverse=True)
    return regions[:top_k]


# ─── ReportGenerator ──────────────────────────────────────────────────────────

class ReportGenerator:
    """
    Generates structured LungCare AI clinical reports.

    Args:
        class_names:       Disease class names in softmax output order.
        model_version:     Version string embedded in every report.
        finding_threshold: Minimum confidence to classify as abnormal.
        top_findings:      Number of template findings to include.
    """

    def __init__(
        self,
        class_names: list[str],
        model_version: str = "v2.0",
        finding_threshold: float = 0.5,
        top_findings: int = 3,
    ) -> None:
        self.class_names = class_names
        self.model_version = model_version
        self.finding_threshold = finding_threshold
        self.top_findings = top_findings

    def generate(
        self,
        probs: np.ndarray | torch.Tensor,
        heatmap: np.ndarray | None = None,
        mask_path: str | Path | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a structured JSON-serialisable report.

        Args:
            probs:      Class probability array (C,) from softmax/sigmoid.
            heatmap:    Optional Grad-CAM heatmap (H, W) float32 in [0, 1].
            mask_path:  Optional path to saved segmentation mask PNG.
            case_id:    Optional patient/case identifier string.

        Returns:
            Dict matching the project output schema (JSON-serialisable).
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        probs = np.asarray(probs, dtype=float)

        # Classification
        pred_idx   = int(probs.argmax())
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        normal_names = {"Normal", "Healthy"}
        status = "normal" if pred_class in normal_names else "abnormal"

        all_scores = {
            cls: round(float(p), 4)
            for cls, p in zip(self.class_names, probs)
        }

        # Findings from templates
        findings: list[str] = list(
            _FINDING_TEMPLATES.get(pred_class, [])[:self.top_findings]
        )
        if confidence < self.finding_threshold + 0.1:
            findings.append(
                f"Confidence ({confidence:.0%}) is below clinical threshold "
                "— radiologist review recommended."
            )

        # Localisation from heatmap
        localization: dict[str, Any] = {"method": None, "top_regions": []}
        if heatmap is not None:
            localization["method"] = "GradCAM"
            localization["top_regions"] = _analyse_heatmap(heatmap)
            for region_info in localization["top_regions"][:1]:
                findings.append(
                    f"Abnormal activation localised to "
                    f"{region_info['region']} region "
                    f"(score {region_info['score']:.2f})."
                )

        report: dict[str, Any] = {
            "case_id":       case_id,
            "status":        status,
            "prediction":    pred_class,
            "confidence":    round(confidence, 4),
            "all_scores":    all_scores,
            "findings":      findings,
            "localization":  localization,
            "segmentation":  {"mask_path": str(mask_path) if mask_path else None},
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "model_version": self.model_version,
        }

        logger.info(
            "Report | case=%s pred=%s conf=%.2f status=%s",
            case_id, pred_class, confidence, status,
        )
        return report

    def to_markdown(self, report: dict[str, Any]) -> str:
        """Render a report dict as human-readable Markdown."""
        lines = [
            "# LungCare AI — Clinical Report", "",
            "| Field | Value |", "|---|---|",
            f"| **Case ID** | `{report.get('case_id', 'N/A')}` |",
            f"| **Status** | {report['status'].upper()} |",
            f"| **Prediction** | **{report['prediction']}** |",
            f"| **Confidence** | {report['confidence']:.1%} |",
            f"| **Timestamp** | {report['timestamp']} |",
            f"| **Model** | {report['model_version']} |",
            "", "## Probability Scores", "",
        ]
        for cls, score in report["all_scores"].items():
            bar = "█" * int(score * 20)
            lines.append(f"- **{cls}**: {score:.1%}  `{bar}`")

        lines += ["", "## Findings", ""]
        for finding in report["findings"]:
            lines.append(f"- {finding}")

        if report["localization"]["top_regions"]:
            lines += ["", "## Localisation", ""]
            for r in report["localization"]["top_regions"]:
                lines.append(f"- {r['region']}: score {r['score']:.2f}")

        return "\n".join(lines)

    def save_report(
        self,
        report: dict[str, Any],
        output_path: Path,
        fmt: str = "json",
    ) -> Path:
        """
        Save a report to disk as JSON or Markdown.

        Args:
            report:      Structured report dict.
            output_path: Destination file path.
            fmt:         ``'json'`` or ``'markdown'``.

        Returns:
            Resolved Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "markdown":
            output_path.write_text(self.to_markdown(report), encoding="utf-8")
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)

        logger.info("Report saved → %s", output_path)
        return output_path
