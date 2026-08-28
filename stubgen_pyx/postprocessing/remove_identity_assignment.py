"""Removes identity assignments (e.g., `x = x`). This usually applies to typedefs for C ints that have already been normalized to int."""

from __future__ import annotations

import ast
import logging
from typing import Any

_logger = logging.getLogger(__name__)


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
            and node.targets[0].id == node.value.id
        ):
            _logger.debug("Removed identity assignment: %s", node.targets[0].id)
            return None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """Remove identity assignments (e.g., `x: Something = x`)."""
        if (
            isinstance(node.target, ast.Name)
            and isinstance(node.value, ast.Name)
            and node.target.id == node.value.id
        ):
            _logger.debug("Removed identity assignment: %s", node.target.id)
            return None
        return node
