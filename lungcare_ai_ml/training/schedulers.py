"""
Learning rate scheduler factory for LungCare AI.

Supports all standard PyTorch schedulers plus a custom
:class:`WarmupCosineScheduler` that linearly ramps the LR from 0 to
``base_lr`` over *warmup_epochs*, then decays with cosine annealing.

Available scheduler names (case-insensitive)
--------------------------------------------
``'cosine'``              → :class:`CosineAnnealingLR`
``'cosine_warm_restarts'``→ :class:`CosineAnnealingWarmRestarts`
``'warmup_cosine'``       → :class:`WarmupCosineScheduler`
``'onecycle'``            → :class:`OneCycleLR`
``'reduce_on_plateau'``   → :class:`ReduceLROnPlateau`
``'step'``                → :class:`StepLR`
``'multistep'``           → :class:`MultiStepLR`
``'linear'``              → :class:`LinearLR`
``'constant'``            → :class:`ConstantLR` (no-op)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.optim as optim
from torch.optim import lr_scheduler

logger = logging.getLogger("lungcare.training.schedulers")


# ─── Custom scheduler ─────────────────────────────────────────────────────────


class WarmupCosineScheduler(lr_scheduler.LambdaLR):
    """
    Linear warmup for *warmup_epochs* then cosine annealing to *eta_min_ratio*.

    The multiplier follows:

    .. math::
        \\lambda(e) = \\begin{cases}
            e / w & e < w \\\\
            r + \\frac{1 - r}{2}\\left(1 + \\cos\\!\\left(\\pi \\cdot \\frac{e-w}{T-w}\\right)\\right) & e \\ge w
        \\end{cases}

    where *w* = warmup_epochs, *T* = total_epochs, *r* = eta_min_ratio.

    Args:
        optimizer: PyTorch optimizer.
        warmup_epochs: Number of linear warmup epochs.
        total_epochs: Total training epochs.
        eta_min_ratio: Final LR as a fraction of base LR (default 0).
        last_epoch: Starting epoch index for resume.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min_ratio = eta_min_ratio

        def _lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return max(epoch / max(warmup_epochs, 1), 1e-6)
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

        super().__init__(optimizer, lr_lambda=_lr_lambda, last_epoch=last_epoch)


# ─── Factory ──────────────────────────────────────────────────────────────────


def build_scheduler(
    optimizer: optim.Optimizer,
    config: Any,
    *,
    total_steps: int | None = None,
    total_epochs: int | None = None,
) -> lr_scheduler.LRScheduler | None:
    """
    Instantiate an LR scheduler from a config object.

    The config object (Pydantic model or ``SimpleNamespace``) must have a
    ``name`` field.  All other fields are optional and fall back to
    sensible defaults.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        config: Scheduler config with at minimum a ``name`` attribute.
        total_steps: Required for ``'onecycle'``.
        total_epochs: Required for ``'warmup_cosine'``.

    Returns:
        An :class:`~torch.optim.lr_scheduler.LRScheduler` instance, or
        ``None`` if ``config.name == 'constant'``.

    Raises:
        ValueError: For unknown scheduler names.
    """

    def _get(attr: str, default: Any = None) -> Any:
        return getattr(config, attr, default)

    name = _get("name", "cosine").lower()
    logger.info("Building scheduler: '%s'", name)

    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=_get("T_max", total_epochs or 100),
            eta_min=_get("eta_min", 1e-6),
        )

    elif name == "cosine_warm_restarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=_get("T_0", 10),
            T_mult=_get("T_mult", 2),
            eta_min=_get("eta_min", 1e-6),
        )

    elif name == "warmup_cosine":
        if total_epochs is None:
            raise ValueError("'warmup_cosine' requires total_epochs argument.")
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=_get("warmup_epochs", 5),
            total_epochs=total_epochs,
            eta_min_ratio=_get("eta_min_ratio", 0.0),
        )

    elif name == "onecycle":
        if total_steps is None:
            raise ValueError("'onecycle' requires total_steps argument.")
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=_get("max_lr", _get("lr", 1e-3)),
            total_steps=total_steps,
            pct_start=_get("pct_start", 0.3),
            div_factor=_get("div_factor", 25.0),
            final_div_factor=_get("final_div_factor", 1e4),
        )

    elif name in ("reduce_on_plateau", "plateau"):
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=_get("mode", "max"),
            patience=_get("patience", 10),
            factor=_get("factor", 0.5),
            min_lr=_get("eta_min", 1e-7),
            threshold=_get("threshold", 1e-4),
        )

    elif name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=_get("step_size", 30),
            gamma=_get("gamma", 0.1),
        )

    elif name == "multistep":
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=_get("milestones", [30, 60, 90]),
            gamma=_get("gamma", 0.1),
        )

    elif name == "linear":
        return lr_scheduler.LinearLR(
            optimizer,
            start_factor=_get("start_factor", 1.0),
            end_factor=_get("end_factor", 0.1),
            total_iters=_get("total_iters", total_epochs or 100),
        )

    elif name in ("constant", "none"):
        return None

    else:
        raise ValueError(
            f"Unknown scheduler '{name}'. "
            "Valid options: cosine, cosine_warm_restarts, warmup_cosine, "
            "onecycle, reduce_on_plateau, step, multistep, linear, constant."
        )


def is_step_scheduler(scheduler: lr_scheduler.LRScheduler | None) -> bool:
    """
    Return ``True`` if the scheduler should step per **batch** (not epoch).

    Only :class:`OneCycleLR` requires per-step calling.
    """
    return isinstance(scheduler, lr_scheduler.OneCycleLR)
