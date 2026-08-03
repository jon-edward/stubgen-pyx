"""Removes the implementation that accompanies ``@overload`` declarations.

In a regular module, a series of ``@overload`` declarations must be followed by
exactly one undecorated implementation, so a ``.pyx`` source has to provide it.
Stub files are exempt from that rule, and mypy rejects a stub that carries the
implementation::

    example.pyi:11: error: An implementation for an overloaded function is not
    allowed in a stub file  [misc]

The declarations describe the entire signature, so the implementation adds
nothing to the stub. mypy's own ``stubgen`` drops it as well, docstring
included.
"""

from __future__ import annotations

import ast

_FUNCTION_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)


def remove_overload_implementations(tree: ast.AST) -> ast.AST:
    """Remove implementations of ``@overload``-declared functions from an AST.

    A definition is removed when it carries no ``@overload`` decorator and at
    least one same-named ``@overload`` definition is present in the same scope.
    The scopes considered are the module body and each class body, so an
    unrelated function of the same name in another scope is never affected.

    Args:
        tree: The AST to process.

    Returns:
        Transformed AST without overload implementations.
    """
    return _OverloadImplementationRemover().visit(tree)


class _OverloadImplementationRemover(ast.NodeTransformer):
    """Removes implementations that accompany ``@overload`` declarations."""

    def visit_Module(self, node: ast.Module) -> ast.AST:
        """Process the module scope."""
        return self._process_scope(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        """Process a class scope."""
        return self._process_scope(node)

    def _process_scope(self, node: ast.Module | ast.ClassDef) -> ast.AST:
        """Remove overload implementations from a single scope."""
        self.generic_visit(node)  # Nested classes hold a scope of their own.

        overloaded = {
            stmt.name
            for stmt in node.body
            if isinstance(stmt, _FUNCTION_DEFS) and _has_overload_decorator(stmt)
        }
        if not overloaded:
            return node

        # A declaration always survives, so the body cannot become empty.
        node.body = [
            stmt
            for stmt in node.body
            if not (
                isinstance(stmt, _FUNCTION_DEFS)
                and stmt.name in overloaded
                and not _has_overload_decorator(stmt)
            )
        ]
        return node


def _has_overload_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Report whether a definition carries an ``@overload`` decorator."""
    return any(_is_overload(decorator) for decorator in node.decorator_list)


def _is_overload(node: ast.expr) -> bool:
    """Recognize ``overload`` and dotted spellings such as ``typing.overload``."""
    return _dotted_name(node).split(".")[-1] == "overload"


def _dotted_name(node: ast.expr) -> str:
    """Return the dotted name of an expression, or ``""`` if it has none."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
