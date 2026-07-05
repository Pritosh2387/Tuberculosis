"""
Healthy reference scan service for LungCare AI.

Maintains a corpus of healthy chest X-ray feature vectors (extracted from
a classifier's GAP layer) and provides:

1. :class:`HealthyReferenceDatabase` — in-memory feature store with
   persistence (numpy ``.npz``).
2. :meth:`extract_features` — extract a GAP embedding from any
   :class:`BaseClassifier` instance.
3. :meth:`get_closest_healthy` — retrieve the K most similar healthy
   scans for visual / feature-space comparison.
4. :meth:`compute_deviation` — normalised L2 / cosine deviation of a
   patient scan from the healthy distribution mean.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("lungcare.services.healthy_reference")


# ─── Feature extraction ───────────────────────────────────────────────────────


@torch.no_grad()
def extract_features(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """
    Extract a GAP (Global Average Pooling) feature vector from a classifier.

    The model must implement :meth:`get_features` which returns the spatial
    feature map ``(1, C, H, W)`` before pooling.  The function applies GAP
    internally.

    Args:
        model: Classifier with ``get_features()`` method.
        input_tensor: Preprocessed image ``(1, C, H, W)``.
        device: Target device.

    Returns:
        1-D float32 numpy array of shape ``(C,)``.

    Raises:
        AttributeError: If the model does not implement ``get_features``.
    """
    if not hasattr(model, "get_features"):
        raise AttributeError(
            "Model must implement get_features() → (1, C, H, W) spatial map."
        )

    model.eval()
    model.to(device)
    inp = input_tensor.to(device)

    spatial_map: torch.Tensor = model.get_features(inp)   # (1, C, H, W)
    gap: torch.Tensor = spatial_map.mean(dim=(2, 3))      # (1, C)
    return gap.squeeze(0).cpu().numpy().astype(np.float32)


# ─── Database ─────────────────────────────────────────────────────────────────


class HealthyReferenceDatabase:
    """
    In-memory store of healthy scan feature embeddings.

    Supports incremental addition of features, nearest-neighbour lookup,
    and mean-deviation scoring.  All features are L2-normalised before
    storage so that cosine similarity ≡ dot product.

    Args:
        dim: Feature dimensionality (e.g. 2048 for ResNet50 GAP).
        metric: Distance metric for nearest-neighbour search.
            ``'cosine'`` or ``'l2'``.
        save_path: Optional file path for persistence (``.npz``).
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        save_path: Path | None = None,
    ) -> None:
        self.dim = dim
        self.metric = metric
        self.save_path = save_path
        self._features: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._meta: list[dict[str, Any]] = []
        self._mean: np.ndarray | None = None   # Cached centroid

    # ─── Population ──────────────────────────────────────────────────────────

    def add(
        self,
        feature: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a single L2-normalised feature vector to the database.

        Args:
            feature: 1-D float32 array of shape ``(dim,)``.
            metadata: Optional dict stored alongside the feature (e.g.
                patient ID, dataset name, scan date).
        """
        feat = feature.astype(np.float32).flatten()
        if feat.shape[0] != self.dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.dim}, got {feat.shape[0]}."
            )
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        self._features = np.vstack([self._features, feat[None]])
        self._meta.append(metadata or {})
        self._mean = None   # Invalidate cached mean

    def add_batch(
        self,
        features: np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Add multiple feature vectors at once.

        Args:
            features: ``(N, dim)`` float32 array.
            metadata: Optional list of N metadata dicts.
        """
        feats = features.astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats = feats / norms
        self._features = np.vstack([self._features, feats])
        meta_list = metadata or [{} for _ in range(len(feats))]
        self._meta.extend(meta_list)
        self._mean = None

    # ─── Querying ─────────────────────────────────────────────────────────────

    def get_mean(self) -> np.ndarray:
        """
        Return (and cache) the centroid of the healthy distribution.

        Returns:
            L2-normalised mean feature vector ``(dim,)``.
        """
        if self._mean is None:
            if len(self._features) == 0:
                raise RuntimeError("Database is empty.")
            raw_mean = self._features.mean(axis=0)
            self._mean = raw_mean / (np.linalg.norm(raw_mean) + 1e-8)
        return self._mean

    def compute_deviation(self, feature: np.ndarray) -> float:
        """
        Compute deviation of *feature* from the healthy distribution centroid.

        Returns a score in roughly [0, 2]:
        - ≈ 0   : near-identical to healthy mean.
        - ≈ 1   : moderate difference.
        - ≈ 2   : maximally dissimilar (opposite cosine direction).

        Args:
            feature: Patient scan feature ``(dim,)`` float32.

        Returns:
            Scalar deviation score (L2 distance in normalised space ≡
            angular distance).
        """
        feat = feature.astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        mean = self.get_mean()
        return float(np.linalg.norm(feat - mean))

    def get_top_k_similar(
        self,
        feature: np.ndarray,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the K most similar healthy scans to *feature*.

        Args:
            feature: Query feature vector ``(dim,)`` float32.
            k: Number of nearest neighbours to retrieve.

        Returns:
            List of dicts with keys ``'rank'``, ``'score'``, ``'metadata'``,
            sorted by similarity (highest first).
        """
        if len(self._features) == 0:
            raise RuntimeError("Database is empty.")

        feat = feature.astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-8)

        if self.metric == "cosine":
            scores = self._features @ feat       # (N,) dot products
        else:
            dists = np.linalg.norm(self._features - feat, axis=1)
            scores = -dists                       # Negate so higher = closer

        k = min(k, len(self._features))
        top_k_idx = np.argsort(scores)[::-1][:k]

        return [
            {
                "rank": i + 1,
                "score": float(scores[idx]),
                "metadata": self._meta[idx],
            }
            for i, idx in enumerate(top_k_idx)
        ]

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """
        Save the database to a compressed ``.npz`` file.

        Args:
            path: Destination path.  Defaults to :attr:`save_path`.

        Returns:
            Resolved save path.
        """
        import json

        dest = Path(path or self.save_path)
        if dest is None:
            raise ValueError("No save path specified.")
        dest.parent.mkdir(parents=True, exist_ok=True)

        meta_bytes = json.dumps(self._meta).encode()
        np.savez_compressed(
            dest,
            features=self._features,
            meta_json=np.frombuffer(meta_bytes, dtype=np.uint8),
            dim=np.array([self.dim]),
        )
        logger.info(
            "Healthy DB saved: %d vectors → %s", len(self._features), dest
        )
        return dest

    @classmethod
    def load(cls, path: Path, metric: str = "cosine") -> "HealthyReferenceDatabase":
        """
        Load a database from a ``.npz`` file.

        Args:
            path: Source file created by :meth:`save`.
            metric: Distance metric to use.

        Returns:
            Populated :class:`HealthyReferenceDatabase` instance.
        """
        import json

        data = np.load(path, allow_pickle=False)
        dim = int(data["dim"][0])
        db = cls(dim=dim, metric=metric, save_path=path)
        db._features = data["features"].astype(np.float32)
        meta_bytes = data["meta_json"].tobytes()
        db._meta = json.loads(meta_bytes.decode())
        logger.info(
            "Healthy DB loaded: %d vectors from %s", len(db._features), path
        )
        return db

    def __len__(self) -> int:
        return len(self._features)

    def __repr__(self) -> str:
        return (
            f"HealthyReferenceDatabase("
            f"n={len(self._features)}, dim={self.dim}, metric='{self.metric}')"
        )
