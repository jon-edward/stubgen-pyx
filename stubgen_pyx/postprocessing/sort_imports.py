"""Sorts imports in Python modules."""

from __future__ import annotations

import isort


def sort_imports(source: str) -> str:
    """Sort imports using the isort tool."""
    return isort.code(source)
