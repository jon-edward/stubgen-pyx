"""Source extraction helpers for Cython AST nodes."""

from __future__ import annotations

import textwrap
from Cython.Compiler import Nodes, ExprNodes


def get_source(source: str, node: Nodes.Node) -> str:
    """Extract source code for a node, dedented and stripped.

    ``end_pos`` is often inaccurate in Cython's AST; the function falls back
    to the start position when it is missing.
    """
    lines = source.splitlines(keepends=True)
    end_pos = node.end_pos() or node.pos
    output = "".join(lines[i - 1] for i in range(node.pos[1], end_pos[1] + 1))
    return textwrap.dedent(output).rstrip()


def get_decorators(
    source: str,
    node: Nodes.DefNode
    | Nodes.CFuncDefNode
    | Nodes.CClassDefNode
    | Nodes.PyClassDefNode,
) -> list[str]:
    """Return decorator source strings for a function or class node."""
    if node.decorators:
        return [get_source(source, d) for d in node.decorators]
    return []


def get_bases(node: Nodes.CClassDefNode | Nodes.PyClassDefNode) -> list[str]:
    """Return base-class name strings from a class node."""
    if not hasattr(node, "bases") or not node.bases:
        return []
    return [
        base.name for base in node.bases.args if isinstance(base, ExprNodes.NameNode)
    ]


def get_metaclass(node: Nodes.PyClassDefNode | Nodes.CClassDefNode) -> str | None:
    """Return the metaclass name from a Python class node, if present."""
    if not isinstance(node, Nodes.PyClassDefNode):
        return None
    if node.metaclass and isinstance(node.metaclass, ExprNodes.NameNode):
        return node.metaclass.name
    return None
