"""
Trim unused imports from a Python AST.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


_RESERVED_MODULES = {"__future__", "asyncio"}


def trim_imports(tree: ast.AST, used_names: set[str]) -> ast.AST:
    """
    Trim unused imports from a Python AST.
    """
    return _UnusedImportRemover(used_names).visit(tree)


@dataclass
class _UnusedImportRemover(ast.NodeTransformer):
    """
    Removes unused imports from a Python AST given a set of used names.

    This transformer identifies and removes:
    - Unused `import` statements
    - Unused names from `from ... import ...` statements
    - Entire import statements if all imported names are unused
    """

    used_names: set[str]

    def visit_Import(self, node: ast.Import) -> ast.Import | None:
        """Remove unused simple imports (e.g., `import foo`)"""
        new_names = []
        for alias in node.names:
            imported_name = alias.asname if alias.asname else alias.name
            if imported_name in self.used_names:
                new_names.append(alias)

        # If no imports remain, remove the entire statement
        if not new_names:
            return None

        # Update the import with only used names
        node.names = new_names
        return node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom | None:
        """Remove unused from-imports (e.g., `from foo import bar`)"""
        # Handle `from foo import *` - we can't know what's used
        if any(alias.name == "*" for alias in node.names):
            return node

        new_names = []

        for alias in node.names:
            # The name that's available in the namespace
            imported_name = alias.asname if alias.asname else alias.name

            if imported_name in self.used_names or node.module in _RESERVED_MODULES:
                new_names.append(alias)

        # If no imports remain, remove the entire statement
        if not new_names:
            return None

        # Update the import with only used names
        node.names = new_names
        return node
