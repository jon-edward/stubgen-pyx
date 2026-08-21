"""Expression unparse helpers for conversion."""

from __future__ import annotations

from typing import ClassVar

from Cython.CodeWriter import ExpressionWriter
from Cython.Compiler import Nodes


class _Unparser(ExpressionWriter):
    """
    stubgen-pyx-specific expression unparser for Cython code.

    This tracks a subset of Cython unary operators that hold no inherent
    meaning in Python and should not be emitted into stubs.
    """

    _unop_ignores: ClassVar[frozenset[str]] = frozenset({"&"})
    _default_unop_precedence: ClassVar[int] = ExpressionWriter.unop_precedence[
        "~"
    ]  # same precedence as `-` and `+`

    def visit_UnopNode(self, node):
        op = node.operator

        if op in self._unop_ignores:
            # Ignored operators (e.g., &) use default precedence but aren't emitted
            prec = self._default_unop_precedence
            should_emit = False
        else:
            # Standard operators must be in the precedence table
            prec = self.unop_precedence[op]
            should_emit = True

        self.operator_enter(prec)
        if should_emit:
            self.put(f"{op}")
        self.visit(node.operand)
        self.operator_exit()

    def emit_sequence(self, node, parens=("", "")):
        open_paren, close_paren = parens
        items = node.subexpr_nodes()
        self.put(open_paren)
        self.comma_separated_list(items)
        if open_paren == "(" and len(items) == 1:
            self.put(",")
        self.put(close_paren)


def unparse_expr(node: Nodes.Node | None) -> str | None:
    """Render an expression node to source code."""
    if node is None:
        return None

    expr_writer = _Unparser(allow_unknown_nodes=True)
    expr_writer.visit(node)
    return expr_writer.result
