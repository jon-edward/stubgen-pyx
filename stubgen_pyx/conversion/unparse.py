"""Expression unparse helpers for conversion."""

from __future__ import annotations

from Cython.CodeWriter import ExpressionWriter
from Cython.Compiler import Nodes


def unparse_expr(node: Nodes.Node | None) -> str | None:
    """Render an expression node to source code."""
    if node is None:
        return None

    expr_writer = ExpressionWriter(allow_unknown_nodes=True)
    expr_writer.visit(node)
    return expr_writer.result
