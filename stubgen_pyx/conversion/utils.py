"""Shared conversion utilities with no Cython-node dependency of their own.

Kept separate from any single conversion module specifically so a small,
generic helper needed by two mutually-importing modules (e.g. `signature.py`
and `type_parsing.py`) has a common, dependency-free home instead of forcing
a circular import between them.
"""

from __future__ import annotations


def decode_or_pass(value: str | bytes) -> str:
    """Ensure value is a string, decoding bytes if needed."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected str or bytes, got {type(value)}")
