"""
Shared postprocessing utilities.
"""

from __future__ import annotations

import ast
import builtins

# Public builtin names (``dir(builtins)`` minus dunders), shared by every pass
# that needs to decide whether a bare name is always resolvable. Passes that
# need extra names on top of this (e.g. module dunders like ``__name__``)
# should union their own additions in, rather than recomputing this set.
PUBLIC_BUILTIN_NAMES: frozenset[str] = frozenset(
    name for name in dir(builtins) if not name.startswith("_")
)


def dotted_name(node: ast.expr) -> str:
    """Return the dotted name of an expression, or ``""`` if it has none."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
