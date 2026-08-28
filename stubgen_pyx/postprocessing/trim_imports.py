"""Removes unused imports from Python .pyi files."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

_RESERVED_MODULES = {"asyncio"}


def trim_imports(tree: ast.AST, used_names: set[str]) -> ast.AST:
    """Remove imports that don't reference any used names."""
    return _UnusedImportRemover(used_names).visit(tree)


@dataclass
class _UnusedImportRemover(ast.NodeTransformer):
    """Remove unused imports from an AST given used names.

    Removes unused `import` and `from ... import ...` statements.
    Removes individual imports if all names in a statement are unused.
    """

    used_names: set[str]

    def _include_alias(self, alias: ast.alias) -> bool:
        """Defines behavior for if an import alias should be included in resulting AST"""
        if alias.asname == alias.name:
            # "X as X" should always be included
            return True
        imported_name = alias.asname or alias.name
        return imported_name in self.used_names or alias.name in _RESERVED_MODULES

    def visit_Import(self, node: ast.Import) -> ast.Import | None:
        """Remove unused simple imports (e.g., `import foo`)."""
        new_names = []
        for alias in node.names:
            if self._include_alias(alias):
                new_names.append(alias)
            else:
                _logger.debug("Removed unused import: %s", alias.name)

        if not new_names:
            return None

        node.names = new_names
        return node

    @staticmethod
    def _import_from_full_name(node: ast.ImportFrom, alias: ast.alias) -> str:
        return (
            f"{'.' * node.level}{node.module + '.' if node.module else ''}{alias.name}"
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        """Remove unused from-imports (e.g., `from foo import bar`)."""
        if any(alias.name == "*" for alias in node.names):
            return node

        new_names = []
        for alias in node.names:
            if self._include_alias(alias):
                new_names.append(alias)
            else:
                _logger.debug(
                    "Removed unused import: %s",
                    self._import_from_full_name(node, alias),
                )

        if not new_names:
            return None

        node.names = new_names
        return node
