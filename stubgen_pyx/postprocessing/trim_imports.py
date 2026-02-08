"""Removes unused imports from Python .pyi files."""

from __future__ import annotations

import ast
from dataclasses import dataclass


_RESERVED_MODULES = {"__future__", "asyncio"}


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

    def visit_Import(self, node: ast.Import) -> ast.Import | None:
        """Remove unused simple imports (e.g., `import foo`)."""
        new_names = []
        for alias in node.names:
            imported_name = alias.asname if alias.asname else alias.name
            if imported_name in self.used_names:
                new_names.append(alias)

        if not new_names:
            return None

        node.names = new_names
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        """Remove unused from-imports (e.g., `from foo import bar`)."""
        if any(alias.name == "*" for alias in node.names):
            return node

        new_names = []

        for alias in node.names:
            imported_name = alias.asname if alias.asname else alias.name

            if imported_name in self.used_names or node.module in _RESERVED_MODULES:
                new_names.append(alias)

        if not new_names:
            return None

        node.names = new_names
        return node
