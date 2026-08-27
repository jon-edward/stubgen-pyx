"""
Logging helpers.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar("T")


def with_debug_fallback(
    value: T | None, fallback: T, message: str | Callable[[], str]
) -> T:
    """Return the value if it is not None, otherwise log a debug message and return the fallback."""
    if value is not None:
        return value
    if message:
        _logger.debug(message() if callable(message) else message)
    return fallback
