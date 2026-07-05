"""
Reproducibility utilities for LungCare AI.

Centralises seed management across Python's ``random``, NumPy, PyTorch
CPU/CUDA, and CUDA deterministic algorithm settings.  Also provides a
``worker_init_fn`` for reproducible multi-process data loading.
"""

import logging
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch

logger = logging.getLogger("lungcare.seed")


@dataclass
class SeedState:
    """
    Snapshot of all random-generator states.

    Captures the full RNG state of Python's ``random``, NumPy, PyTorch
    CPU, and all CUDA devices at a given moment so training can be
    resumed byte-for-byte identically.
    """

    python_state: tuple
    numpy_state: dict
    torch_state: torch.Tensor
    cuda_states: list[torch.Tensor] = field(default_factory=list)


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """
    Set seeds for all RNG sources used by the project.

    Covers Python ``random``, ``PYTHONHASHSEED``, NumPy, PyTorch (CPU),
    and all available CUDA devices.

    Args:
        seed: Integer seed. Must be in ``[0, 2**32 - 1]``.
        deterministic: When ``True``, enables ``cuDNN`` deterministic mode
            and ``torch.use_deterministic_algorithms``.  This trades
            performance for exact reproducibility.  Has no effect when CUDA
            is unavailable.

    Returns:
        The applied seed value (useful for logging).

    Raises:
        ValueError: If *seed* is outside the valid range.

    Note:
        Even in deterministic mode, floating-point reductions may differ
        across CUDA architectures.  For strict cross-machine reproducibility
        use CPU inference.
    """
    if not (0 <= seed <= 2**32 - 1):
        raise ValueError(
            f"Seed must be in [0, 2**32 - 1], got {seed}."
        )

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            logger.info(
                "CUDA deterministic mode ON — training will be slower."
            )
        else:
            torch.backends.cudnn.benchmark = True
            logger.debug("cuDNN benchmark mode ON (non-deterministic, faster).")

    logger.info("Global seed set to %d (deterministic=%s).", seed, deterministic)
    return seed


def get_seed_state() -> SeedState:
    """
    Capture the current state of all random generators.

    The returned :class:`SeedState` can be serialised with
    ``torch.save`` and later passed to :func:`restore_seed_state` to
    resume from an exact point in a training run.

    Returns:
        A :class:`SeedState` snapshot of all live RNG states.
    """
    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [
            torch.cuda.get_rng_state(device)
            for device in range(torch.cuda.device_count())
        ]

    return SeedState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state(),  # type: ignore[arg-type]
        torch_state=torch.get_rng_state(),
        cuda_states=cuda_states,
    )


def restore_seed_state(state: SeedState) -> None:
    """
    Restore all random generators to a previously captured state.

    Args:
        state: A :class:`SeedState` obtained from :func:`get_seed_state`.

    Note:
        CUDA device count is checked at restoration time; if the number of
        GPUs has changed since the snapshot, only the available devices are
        restored.
    """
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_state)

    if torch.cuda.is_available() and state.cuda_states:
        n_devices = torch.cuda.device_count()
        for device_idx, cuda_state in enumerate(state.cuda_states):
            if device_idx < n_devices:
                torch.cuda.set_rng_state(cuda_state, device_idx)

    logger.debug("Random generator states restored from snapshot.")


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker initialiser for reproducible multi-process loading.

    Derive a per-worker seed from PyTorch's base seed so that each worker
    produces a deterministic but distinct sequence of random numbers.

    Pass this to :class:`torch.utils.data.DataLoader` as the
    ``worker_init_fn`` argument:

    .. code-block:: python

        DataLoader(dataset, worker_init_fn=worker_init_fn)

    Args:
        worker_id: Worker index assigned by the PyTorch DataLoader.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
