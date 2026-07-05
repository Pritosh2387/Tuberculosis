"""
Pytest session bootstrap for LungCare AI.

This module is imported by pytest *before* any test module (and therefore
before ``transformers`` is transitively imported via ``torchvision`` /
``torchmetrics``).  It installs a small forward-compatibility shim so the
test suite runs on the pinned ``torch`` build even when a newer
``transformers`` is present in the environment.

Root cause it addresses
-----------------------
``transformers>=4.4x`` calls ``torch.utils._pytree.register_pytree_node`` at
import time.  That public alias only exists on ``torch>=2.2``; older builds
(e.g. ``torch==2.1.2``) expose the private ``_register_pytree_node`` instead.
Without the alias, importing anything that pulls in ``transformers`` raises
``AttributeError: module 'torch.utils._pytree' has no attribute
'register_pytree_node'``.

The shim is a no-op on torch builds that already provide the public name, so
it is safe across the ``torch>=2.1`` range the project targets.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when tests are run from any CWD.
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Apply the torch/transformers pytree compatibility shim as early as possible.
from utils.torch_compat import ensure_pytree_compat  # noqa: E402

ensure_pytree_compat()
