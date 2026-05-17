"""Removes identity assignments (e.g., `x = x`). This usually applies to typedefs for C ints that have already been normalized to int."""

from __future__ import annotations

import ast


def remove_identity_assignment(tree: ast.AST) -> ast.AST:
    """Remove identity assignments (e.g., `x = x`) from an AST."""
    return _IdentityAssignmentRemover().visit(tree)


class _IdentityAssignmentRemover(ast.NodeTransformer):
    """Removes identity assignments (e.g., `x = x`)."""

    def visit_Assign(self, node: ast.Assign) -> ast.AST | None:
        """Remove identity assignments (e.g., `x = x`)."""
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Name)
        ):
            if node.targets[0].id == node.value.id:
                return None
        return node
