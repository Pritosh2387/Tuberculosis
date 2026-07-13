"""
conftest.py
────────────
Pytest configuration for LungCare AI test suite.

Adds the repository root to sys.path so all imports work from any
working directory without a package install.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
