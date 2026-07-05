"""
Abstract base classifier for LungCare AI.

All classification architectures (ResNet50, DenseNet121, EfficientNet-B0,
Vision Transformer) inherit from :class:`BaseClassifier`, which provides
shared inference utilities, parameter management, and an interface contract
for the explainability modules.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("lungcare.models.classifier")


class BaseClassifier(nn.Module, ABC):
    """
    Abstract base class for all LungCare AI classification models.

    Concrete subclasses must implement:
    - :meth:`forward` (from :class:`nn.Module`)
    - :meth:`get_features` — spatial feature maps before global pooling
    - :meth:`get_target_layer` — the :class:`nn.Module` to hook for Grad-CAM

    Conventions
    -----------
    - The classification head must be stored as ``self.classifier`` so that
      :meth:`freeze_backbone` can unfreeze it selectively.
    - ``forward`` must return **raw logits** (no softmax/sigmoid applied).

    Args:
        num_classes: Number of output classes.
        task: One of ``'binary'``, ``'multiclass'``, ``'multilabel'``.
        dropout_rate: Dropout probability applied in the classification head.
    """

    def __init__(
        self,
        num_classes: int,
        task: str = "multiclass",
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task = task
        self.dropout_rate = dropout_rate

    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps before the global pooling layer.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Feature map tensor of shape ``(B, C', H', W')`` where
            ``H'`` and ``W'`` are spatially reduced relative to the input.
        """

    @abstractmethod
    def get_target_layer(self) -> nn.Module:
        """
        Return the module to register hooks on for Grad-CAM.

        This should be the **last convolutional / dense block** before
        global average pooling.
        """

    # ─── Inference helpers ────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return class probabilities.

        Args:
            x: Input image tensor ``(B, C, H, W)``.

        Returns:
            - ``'binary'`` / ``'multilabel'``: sigmoid probabilities.
            - ``'multiclass'``: softmax probabilities.
        """
        self.eval()
        logits = self.forward(x)
        if self.task in ("binary", "multilabel"):
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_class(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Return predicted class index / binary vector.

        Args:
            x: Input image tensor ``(B, C, H, W)``.
            threshold: Decision threshold for binary / multilabel tasks.

        Returns:
            - ``'multiclass'``: ``(B,)`` long tensor of class indices.
            - ``'binary'`` / ``'multilabel'``: ``(B, num_classes)`` long tensor.
        """
        probs = self.predict(x)
        if self.task == "multiclass":
            return probs.argmax(dim=1)
        return (probs >= threshold).long()

    # ─── Parameter management ─────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Freeze all parameters except those in ``self.classifier``.

        Allows fine-tuning only the classification head while keeping
        the backbone weights fixed.
        """
        for param in self.parameters():
            param.requires_grad = False
        if hasattr(self, "classifier"):
            for param in self.classifier.parameters():
                param.requires_grad = True
        logger.info(
            "%s: backbone frozen. Trainable params: %d",
            self.__class__.__name__,
            self.count_parameters(),
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all model parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info(
            "%s: all parameters unfrozen. Trainable params: %d",
            self.__class__.__name__,
            self.count_parameters(),
        )

    def unfreeze_last_n_layers(self, n: int) -> None:
        """
        Unfreeze the last *n* named parameter groups (progressive unfreezing).

        Args:
            n: Number of named parameter groups to unfreeze from the end.
        """
        all_params = list(self.named_parameters())
        freeze_until = max(0, len(all_params) - n)
        for i, (_, param) in enumerate(all_params):
            param.requires_grad = i >= freeze_until

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Return the parameter count.

        Args:
            trainable_only: When ``True``, count only parameters with
                ``requires_grad=True``.

        Returns:
            Integer parameter count.
        """
        return sum(
            p.numel()
            for p in self.parameters()
            if (p.requires_grad if trainable_only else True)
        )

    # ─── Serialisation ────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: Path | str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Save the model state dict and construction metadata.

        Args:
            path: Destination ``.pth`` file.
            metadata: Optional dict merged into the saved dict.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "num_classes": self.num_classes,
                "task": self.task,
                "dropout_rate": self.dropout_rate,
                "architecture": self.__class__.__name__,
                **(metadata or {}),
            },
            save_path,
        )
        logger.info("Model saved: %s", save_path.name)

    def load_weights(self, path: Path | str, strict: bool = True) -> None:
        """
        Load model weights from a checkpoint file.

        Args:
            path: Path to a ``.pth`` checkpoint file.
            strict: Passed to :meth:`load_state_dict`.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.load_state_dict(state, strict=strict)
        logger.info("Weights loaded from %s (strict=%s).", ckpt_path.name, strict)

    # ─── Unit-test helpers ────────────────────────────────────────────────────

    def get_input_example(
        self,
        batch_size: int = 1,
        channels: int = 3,
        size: int = 224,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """
        Return a zero-filled dummy input tensor for smoke tests.

        Args:
            batch_size: Number of samples in the batch.
            channels: Number of input channels.
            size: Spatial size (height = width).
            device: Target device.

        Returns:
            Float tensor of shape ``(batch_size, channels, size, size)``.
        """
        return torch.zeros(batch_size, channels, size, size, device=device)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"task='{self.task}', "
            f"dropout={self.dropout_rate}, "
            f"params={self.count_parameters():,})"
        )
