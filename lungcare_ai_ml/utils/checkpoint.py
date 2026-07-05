"""
Checkpoint manager for LungCare AI training runs.

Provides top-K checkpoint saving with a JSON manifest file, automatic
best/latest tracking, and clean resume-training support.  All disk I/O
uses :mod:`pathlib` and is platform-independent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("lungcare.checkpoint")

_MANIFEST_FILENAME = ".checkpoint_manifest.json"


@dataclass
class CheckpointEntry:
    """Metadata for a single saved checkpoint file."""

    path: str
    epoch: int
    metrics: dict[str, float]
    timestamp: str
    is_best: bool = False
    is_last: bool = False


@dataclass
class CheckpointManifest:
    """
    Persistent record of all checkpoints for a training run.

    Serialised as ``<checkpoint_dir>/.checkpoint_manifest.json``.
    """

    monitor: str
    mode: str
    top_k: int
    checkpoints: list[CheckpointEntry] = field(default_factory=list)
    best_path: str | None = None
    last_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict."""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "top_k": self.top_k,
            "checkpoints": [asdict(c) for c in self.checkpoints],
            "best_path": self.best_path,
            "last_path": self.last_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointManifest":
        """Deserialise from a JSON-compatible dict."""
        entries = [CheckpointEntry(**c) for c in data.get("checkpoints", [])]
        return cls(
            monitor=data["monitor"],
            mode=data["mode"],
            top_k=data["top_k"],
            checkpoints=entries,
            best_path=data.get("best_path"),
            last_path=data.get("last_path"),
        )


class CheckpointManager:
    """
    Save, prune, and load model checkpoints with a persistent manifest.

    Tracks all checkpoints in a JSON sidecar file.  Supports top-K
    pruning (by monitored metric) while always preserving the *best*
    and *last* checkpoints regardless of the K limit.

    Args:
        checkpoint_dir: Directory where ``.pth`` files are written.
        monitor: Metric key to use for best-model selection
            (must appear in the ``metrics`` dict passed to :meth:`save`).
        mode: ``'min'`` to prefer lower values, ``'max'`` to prefer higher.
        top_k: Maximum number of non-best, non-last checkpoints to retain.

    Raises:
        ValueError: If *mode* is not ``'min'`` or ``'max'``.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        monitor: str = "val_loss",
        mode: str = "min",
        top_k: int = 3,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(
                f"mode must be 'min' or 'max', got '{mode}'."
            )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k

        self._manifest_path = self.checkpoint_dir / _MANIFEST_FILENAME
        self._manifest = self._load_manifest()
        self._is_better = (
            (lambda a, b: a < b) if mode == "min" else (lambda a, b: a > b)
        )

    def _load_manifest(self) -> CheckpointManifest:
        """Load manifest from disk, or create a fresh one if absent/corrupt."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r", encoding="utf-8") as fh:
                    return CheckpointManifest.from_dict(json.load(fh))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning(
                    "Corrupt checkpoint manifest — starting fresh. Cause: %s", exc
                )
        return CheckpointManifest(
            monitor=self.monitor, mode=self.mode, top_k=self.top_k
        )

    def _persist_manifest(self) -> None:
        """Write the manifest to disk atomically via a temp file."""
        tmp = self._manifest_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self._manifest.to_dict(), fh, indent=2)
        tmp.replace(self._manifest_path)

    def _auto_filename(self, epoch: int, metrics: dict[str, float]) -> str:
        """Build a human-readable filename from epoch and monitored metric."""
        val = metrics.get(self.monitor, 0.0)
        return f"epoch_{epoch:03d}-{self.monitor}_{val:.4f}.pth"

    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        metrics: dict[str, float],
        filename: str | None = None,
    ) -> Path:
        """
        Save a checkpoint and update the manifest.

        The *state* dict should contain at minimum:
        ``model_state_dict``, ``optimizer_state_dict``, and optionally
        ``scheduler_state_dict``.  The *epoch* and *metrics* are merged
        into the saved file automatically.

        Args:
            state: Arbitrary state dict to persist (model, optimiser, etc.).
            epoch: Current training epoch (1-based).
            metrics: All metric values for this epoch.
            filename: Custom filename.  Auto-generated if ``None``.

        Returns:
            Absolute :class:`pathlib.Path` to the saved checkpoint.
        """
        fname = filename or self._auto_filename(epoch, metrics)
        save_path = self.checkpoint_dir / fname

        torch.save({**state, "epoch": epoch, "metrics": metrics}, save_path)
        logger.info("Checkpoint saved: %s", save_path.name)

        timestamp = datetime.now(tz=timezone.utc).isoformat()
        entry = CheckpointEntry(
            path=str(save_path),
            epoch=epoch,
            metrics=metrics,
            timestamp=timestamp,
        )

        self._update_best_flag(entry, metrics)

        for prev in self._manifest.checkpoints:
            prev.is_last = False
        entry.is_last = True
        self._manifest.last_path = str(save_path)

        self._manifest.checkpoints.append(entry)
        self._prune()
        self._persist_manifest()
        return save_path

    def _update_best_flag(
        self, entry: CheckpointEntry, metrics: dict[str, float]
    ) -> None:
        """Determine whether *entry* is the new best and update flags."""
        monitor_val = metrics.get(self.monitor)
        if monitor_val is None:
            return

        is_first = self._manifest.best_path is None
        if is_first:
            entry.is_best = True
            self._manifest.best_path = entry.path
            return

        best_entry = next(
            (
                c
                for c in self._manifest.checkpoints
                if c.path == self._manifest.best_path
            ),
            None,
        )
        current_best_val = (
            best_entry.metrics.get(self.monitor) if best_entry else None
        )
        if current_best_val is None or self._is_better(monitor_val, current_best_val):
            for c in self._manifest.checkpoints:
                c.is_best = False
            entry.is_best = True
            self._manifest.best_path = entry.path
            logger.info(
                "New best checkpoint — %s: %.4f", self.monitor, monitor_val
            )

    def _prune(self) -> None:
        """
        Remove checkpoints beyond top_k.

        The *best* and *last* checkpoints are always preserved,
        regardless of their rank.
        """
        protected = [c for c in self._manifest.checkpoints if c.is_best or c.is_last]
        candidates = [
            c for c in self._manifest.checkpoints if not c.is_best and not c.is_last
        ]

        reverse = self.mode == "max"
        candidates.sort(
            key=lambda c: c.metrics.get(self.monitor, float("inf")),
            reverse=reverse,
        )

        keep_n = max(0, self.top_k - len(protected))
        to_keep = candidates[:keep_n]
        to_delete = candidates[keep_n:]

        for entry in to_delete:
            p = Path(entry.path)
            if p.exists():
                p.unlink()
                logger.debug("Pruned checkpoint: %s", p.name)

        self._manifest.checkpoints = protected + to_keep

    def load(self, path: Path | str) -> dict[str, Any]:
        """
        Load a checkpoint from a ``.pth`` file.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The checkpoint dict as saved by :meth:`save`.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint: dict[str, Any] = torch.load(
            load_path, map_location="cpu", weights_only=False
        )
        logger.info(
            "Loaded checkpoint: %s  (epoch=%d)",
            load_path.name,
            checkpoint.get("epoch", -1),
        )
        return checkpoint

    def get_best_checkpoint(self) -> Path | None:
        """
        Return the path to the best checkpoint, or ``None`` if none exists.

        The returned path is verified to exist on disk; if the file has been
        moved or deleted externally, ``None`` is returned.
        """
        if self._manifest.best_path is None:
            return None
        p = Path(self._manifest.best_path)
        return p if p.exists() else None

    def get_latest_checkpoint(self) -> Path | None:
        """
        Return the path to the most recently saved checkpoint.

        Same existence check as :meth:`get_best_checkpoint`.
        """
        if self._manifest.last_path is None:
            return None
        p = Path(self._manifest.last_path)
        return p if p.exists() else None

    def resume(self) -> tuple[dict[str, Any] | None, int]:
        """
        Load the latest checkpoint for training resumption.

        Returns:
            A tuple ``(checkpoint_dict, start_epoch)``.  If no checkpoint
            exists returns ``(None, 0)``.  *start_epoch* is the epoch
            **after** the last saved one (i.e. where training should resume).
        """
        latest = self.get_latest_checkpoint()
        if latest is None:
            logger.info("No checkpoint found — starting training from epoch 1.")
            return None, 0

        ckpt = self.load(latest)
        last_epoch = ckpt.get("epoch", 0)
        logger.info(
            "Resuming training from epoch %d (last completed: %d).",
            last_epoch + 1,
            last_epoch,
        )
        return ckpt, last_epoch

    def list_checkpoints(self) -> list[CheckpointEntry]:
        """Return all tracked :class:`CheckpointEntry` objects."""
        return list(self._manifest.checkpoints)

    def delete_all(self) -> None:
        """
        Delete all checkpoint files and reset the manifest.

        Use with caution — intended for test teardown or explicit cleanup.
        """
        for entry in self._manifest.checkpoints:
            p = Path(entry.path)
            if p.exists():
                p.unlink()

        if self._manifest_path.exists():
            self._manifest_path.unlink()

        self._manifest = CheckpointManifest(
            monitor=self.monitor, mode=self.mode, top_k=self.top_k
        )
        logger.info("All checkpoints deleted and manifest reset.")
