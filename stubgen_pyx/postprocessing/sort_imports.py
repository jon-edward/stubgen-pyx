"""Utility for sorting imports."""

import isort


def sort_imports(source: str) -> str:
    """Sort imports in a Python module."""
    return isort.code(source)
