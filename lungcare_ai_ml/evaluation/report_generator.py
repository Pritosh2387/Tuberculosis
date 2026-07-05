"""
Structured clinical report generator for LungCare AI.

Given model outputs (classification logits + segmentation masks + CAM
heatmaps), generates a machine-readable structured report conforming to
the project's standard JSON schema, plus an optional human-readable
Markdown / HTML summary.

Output schema
-------------
.. code-block:: json

    {
        "status":     "abnormal",
        "prediction": "Tuberculosis",
        "confidence": 0.92,
        "all_scores": {"Healthy": 0.03, "Tuberculosis": 0.92, ...},
        "findings":   ["Opacity in upper-right lobe", ...],
        "localization": {
            "method": "GradCAM",
            "top_regions": [{"region": "upper-right", "score": 0.87}]
        },
        "segmentation": {
            "dice_estimate": null,
            "mask_path":     "results/case_001_mask.png"
        },
        "scan_comparison": {
            "deviation_score": 0.31,
            "comparison_notes": ["Increased opacity vs healthy baseline"]
        },
        "timestamp": "2025-01-15T09:32:00Z",
        "model_version": "resnet50-v1.2"
    }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("lungcare.evaluation.report_generator")


# ─── Finding templates ────────────────────────────────────────────────────────

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
    "Healthy": [
        "No significant abnormality detected",
        "Lung fields appear clear",
        "Normal cardiomediastinal silhouette",
    ],
}

_ANATOMICAL_REGIONS = [
    "upper-right",
    "upper-left",
    "mid-right",
    "mid-left",
    "lower-right",
    "lower-left",
    "perihilar",
    "bilateral",
]


# ─── Heatmap → region analysis ────────────────────────────────────────────────


def _analyse_heatmap(
    heatmap: np.ndarray,
    top_k: int = 3,
    threshold: float = 0.4,
) -> list[dict[str, Any]]:
    """
    Map activation peaks in a heatmap to coarse anatomical regions.

    Divides the heatmap into a 2×3 grid (upper / mid / lower × left / right)
    and returns the ``top_k`` regions with highest mean activation above
    *threshold*.

    Args:
        heatmap: Normalised float32 array ``(H, W)`` in ``[0, 1]``.
        top_k: Maximum number of regions to report.
        threshold: Minimum mean activation for a region to be reported.

    Returns:
        List of ``{'region': str, 'score': float}`` dicts, sorted by score.
    """
    H, W = heatmap.shape
    rows = [(0, H // 3, "upper"), (H // 3, 2 * H // 3, "mid"), (2 * H // 3, H, "lower")]
    cols = [(0, W // 2, "right"), (W // 2, W, "left")]

    regions: list[dict[str, Any]] = []
    for r0, r1, row_name in rows:
        for c0, c1, col_name in cols:
            patch = heatmap[r0:r1, c0:c1]
            score = float(patch.mean())
            if score >= threshold:
                regions.append({"region": f"{row_name}-{col_name}", "score": round(score, 3)})

    regions.sort(key=lambda x: x["score"], reverse=True)
    return regions[:top_k]


# ─── Healthy comparison ───────────────────────────────────────────────────────


def _compare_to_healthy(
    pred_features: np.ndarray | None,
    healthy_features: np.ndarray | None,
) -> dict[str, Any]:
    """
    Compute a deviation score between patient and healthy reference features.

    Uses normalised L2 distance as a simple proxy for pathology severity.

    Args:
        pred_features: Feature vector from the patient scan.
        healthy_features: Feature vector from a healthy reference.

    Returns:
        Dict with ``'deviation_score'`` and ``'comparison_notes'``.
    """
    if pred_features is None or healthy_features is None:
        return {"deviation_score": None, "comparison_notes": []}

    pred_n = pred_features / (np.linalg.norm(pred_features) + 1e-8)
    ref_n = healthy_features / (np.linalg.norm(healthy_features) + 1e-8)
    deviation = float(np.linalg.norm(pred_n - ref_n))

    notes: list[str] = []
    if deviation > 0.8:
        notes.append("Significant deviation from healthy reference baseline")
    elif deviation > 0.4:
        notes.append("Moderate deviation from healthy reference — clinical review advised")
    else:
        notes.append("Findings close to healthy reference baseline")

    return {"deviation_score": round(deviation, 4), "comparison_notes": notes}


# ─── Main report generator ────────────────────────────────────────────────────


class ReportGenerator:
    """
    Generates structured LungCare AI clinical reports.

    Args:
        class_names: Disease class names in softmax output order.
        model_version: Version string embedded in every report.
        finding_threshold: Minimum confidence for classifying as abnormal.
        top_findings: Number of template findings to include per prediction.
    """

    def __init__(
        self,
        class_names: list[str],
        model_version: str = "v1.0",
        finding_threshold: float = 0.5,
        top_findings: int = 3,
    ) -> None:
        self.class_names = class_names
        self.model_version = model_version
        self.finding_threshold = finding_threshold
        self.top_findings = top_findings

    # ─── Primary API ─────────────────────────────────────────────────────────

    def generate(
        self,
        probs: np.ndarray | torch.Tensor,
        heatmap: np.ndarray | None = None,
        mask_path: str | Path | None = None,
        healthy_features: np.ndarray | None = None,
        patient_features: np.ndarray | None = None,
        case_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a structured JSON-serialisable report.

        Args:
            probs: Class probability array ``(C,)`` (softmax or sigmoid).
            heatmap: Normalised CAM / Grad-CAM heatmap ``(H, W)`` in [0,1].
            mask_path: Path to saved segmentation mask PNG.
            healthy_features: GAP feature vector from a healthy reference scan.
            patient_features: GAP feature vector from the patient scan.
            case_id: Optional patient / case identifier.

        Returns:
            Structured report dict matching the project's output schema.
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        probs = probs.astype(float)

        # ── Classification ────────────────────────────────────────────────────
        pred_idx = int(probs.argmax())
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])
        status = "normal" if pred_class == "Healthy" else "abnormal"

        all_scores = {
            cls: round(float(p), 4)
            for cls, p in zip(self.class_names, probs)
        }

        # ── Findings ──────────────────────────────────────────────────────────
        templates = _FINDING_TEMPLATES.get(pred_class, [])
        findings = templates[: self.top_findings]

        # If low confidence, append uncertainty note
        if confidence < self.finding_threshold + 0.1:
            findings.append(
                f"Confidence ({confidence:.0%}) below clinical threshold "
                "— radiologist review recommended."
            )

        # ── Localisation ──────────────────────────────────────────────────────
        localization: dict[str, Any] = {"method": None, "top_regions": []}
        if heatmap is not None:
            localization["method"] = "GradCAM"
            localization["top_regions"] = _analyse_heatmap(heatmap)
            # Append region-specific finding
            for region_info in localization["top_regions"][:1]:
                findings.append(
                    f"Abnormal activation localised to {region_info['region']} region "
                    f"(activation score {region_info['score']:.2f})."
                )

        # ── Segmentation ──────────────────────────────────────────────────────
        segmentation: dict[str, Any] = {
            "mask_path": str(mask_path) if mask_path else None,
        }

        # ── Comparison with healthy scan ──────────────────────────────────────
        scan_comparison = _compare_to_healthy(patient_features, healthy_features)

        # ── Assemble ──────────────────────────────────────────────────────────
        report: dict[str, Any] = {
            "case_id": case_id,
            "status": status,
            "prediction": pred_class,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "findings": findings,
            "localization": localization,
            "segmentation": segmentation,
            "scan_comparison": scan_comparison,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": self.model_version,
        }

        logger.info(
            "Report generated | case=%s | pred=%s | conf=%.2f | status=%s",
            case_id, pred_class, confidence, status,
        )
        return report

    # ─── Output formatters ────────────────────────────────────────────────────

    def to_markdown(self, report: dict[str, Any]) -> str:
        """
        Render a structured report as a human-readable Markdown string.

        Args:
            report: Dict returned by :meth:`generate`.

        Returns:
            Multi-line Markdown string.
        """
        lines: list[str] = [
            f"# LungCare AI — Clinical Report",
            f"",
            f"| Field | Value |",
            f"|---|---|",
            f"| **Case ID** | `{report.get('case_id', 'N/A')}` |",
            f"| **Status** | {report['status'].upper()} |",
            f"| **Prediction** | **{report['prediction']}** |",
            f"| **Confidence** | {report['confidence']:.1%} |",
            f"| **Timestamp** | {report['timestamp']} |",
            f"| **Model** | {report['model_version']} |",
            f"",
            f"## Probability Scores",
            f"",
        ]
        for cls, score in report["all_scores"].items():
            bar = "█" * int(score * 20)
            lines.append(f"- **{cls}**: {score:.1%}  `{bar}`")

        lines += [
            f"",
            f"## Findings",
            f"",
        ]
        for finding in report["findings"]:
            lines.append(f"- {finding}")

        if report["localization"]["top_regions"]:
            lines += ["", "## Localisation", ""]
            for r in report["localization"]["top_regions"]:
                lines.append(f"- {r['region']}: score {r['score']:.2f}")

        comp = report["scan_comparison"]
        if comp.get("deviation_score") is not None:
            lines += [
                "",
                "## Comparison with Healthy Baseline",
                "",
                f"- Deviation score: **{comp['deviation_score']:.4f}**",
            ]
            for note in comp["comparison_notes"]:
                lines.append(f"- {note}")

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
            report: Structured report dict.
            output_path: Destination file path (extension determines format
                if ``fmt`` is not specified).
            fmt: ``'json'`` or ``'markdown'``.

        Returns:
            Resolved :class:`Path` to the saved file.
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "markdown":
            output_path.write_text(self.to_markdown(report), encoding="utf-8")
        else:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)

        logger.info("Report saved → %s", output_path)
        return output_path
