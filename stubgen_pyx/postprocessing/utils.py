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


def root_name(node: ast.expr) -> str | None:
    """Return the leftmost ``Name`` id of a dotted attribute/call chain.

    For ``pkg.mod.Type`` and ``pkg.mod.Type(1)`` this returns ``"pkg"``.
    Returns ``None`` if the chain doesn't bottom out in a plain name (e.g. it
    starts with a subscript, literal, or comprehension).

    Used anywhere a pass needs to know whether a dotted or called expression
    is "rooted" in a known name, without caring about the rest of the chain -
    e.g. deciding whether ``some_module.Type`` is resolvable based only on
    whether ``some_module`` is defined.
    """
    while isinstance(node, (ast.Attribute, ast.Call)):
        node = node.func if isinstance(node, ast.Call) else node.value
    return node.id if isinstance(node, ast.Name) else None
