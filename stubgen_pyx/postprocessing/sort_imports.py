"""Utility for sorting imports."""

from __future__ import annotations

import isort


def sort_imports(source: str) -> str:
    """Sort imports in a Python module."""
    return isort.code(source)
