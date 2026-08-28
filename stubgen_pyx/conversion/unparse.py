"""Expression unparse helpers for conversion."""

from __future__ import annotations

import logging
from typing import ClassVar

from Cython.CodeWriter import ExpressionWriter
from Cython.Compiler import Nodes

_logger = logging.getLogger(__name__)


def unparse_expr(node: Nodes.Node | None) -> str | None:
    """Render an expression node to source code."""
    if node is None:
        return None

    expr_writer = _Unparser()
    expr_writer.visit(node)

    if expr_writer.found_unknown is not None:
        # Cython has special handling for errors occuring within
        # a compiler pass, so we can't throw errors from within the visitor.
        # This flag tracks whether an unknown node type was encountered and
        # returns None if so.
        _logger.debug(
            "Unknown node type encountered: %s; Parsed expression as: %r",
            _get_full_class_name(expr_writer.found_unknown),
            expr_writer.result,
        )
        return None

    return expr_writer.result


class _Unparser(ExpressionWriter):
    """
    stubgen-pyx-specific expression unparser for Cython code.

    This tracks a subset of Cython unary operators that hold no inherent
    meaning in Python and should not be emitted into stubs, fixes mono-term
    tuples without trailing comma, and handles nested comprehensions.
    """

    _unop_ignores: ClassVar[frozenset[str]] = frozenset({"&"})
    _default_unop_precedence: ClassVar[int] = ExpressionWriter.unop_precedence[
        "~"
    ]  # same precedence as `-` and `+`

    found_unknown: Nodes.Node | None = None
    """Whether an unknown node type was encountered."""

    def visit_Node(self, node):
        """Generic node visitor. Used for unknown nodes."""
        self.put("...")
        if self.found_unknown:
            # Only track first unknown node
            return
        self.found_unknown = node

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

    def emit_comprehension(self, body, target, sequence, condition, parens=("", "")):
        open_paren, close_paren = parens
        clauses = []
        while isinstance(body, Nodes.ForInStatNode):
            clauses.append((target, sequence, condition))
            target = body.target
            sequence = body.iterator.sequence
            condition = None
            body = body.body
        clauses.append((target, sequence, condition))

        self.put(open_paren)
        self.visit(body)
        for target, sequence, condition in clauses:
            self.put(" for ")
            self.visit(target)
            self.put(" in ")
            self.visit(sequence)
            if condition:
                self.put(" if ")
                self.visit(condition)
        self.put(close_paren)

    def visit_ForInStatNode(self, node):
        self.visit(node.body)
        self.put(" for ")
        if node.target.is_sequence_constructor:
            self.comma_separated_list(node.target.args)
        else:
            self.visit(node.target)
        self.put(" in ")
        self.visit(node.iterator.sequence)


def _get_full_class_name(obj):
    cls = type(obj)
    module = cls.__module__
    qualname = cls.__qualname__
    if module and module not in ("__builtin__", "builtins"):
        return f"{module}.{qualname}"
    return qualname
