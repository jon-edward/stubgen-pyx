"""
Removes all but the last import statement that provides the same name.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def deduplicate_imports(node: ast.AST) -> ast.Module:
    return _DuplicateImportRemover().visit(node)


@dataclass
class _DuplicateImportRemover(ast.NodeTransformer):
    """
    Removes all but the last import statement that provides the same name.

    In Python, if a name is imported multiple times, the last import wins.
    This transformer removes earlier imports of the same name, keeping only
    the final one.
    """

    name_to_imports: Dict[
        str, List[tuple[ast.Import | ast.ImportFrom, int, ast.alias]]
    ] = field(default_factory=dict)

    current_body: Optional[List] = None
    nodes_to_remove: set[ast.AST] = field(default_factory=set)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Process the module, tracking all imports first"""
        self.current_body = node.body

        # First pass: collect all imports and their positions
        for idx, stmt in enumerate(node.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self._register_import(stmt, idx)

        # Determine which imports to remove (all but the last)
        self._mark_duplicates_for_removal()

        # Second pass: remove marked imports
        new_body = []
        for idx, stmt in enumerate(node.body):
            if stmt not in self.nodes_to_remove:
                new_body.append(self.generic_visit(stmt))

        node.body = new_body
        return node

    def _register_import(self, node: ast.Import | ast.ImportFrom, idx: int):
        """Register an import statement and track what names it provides"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                # The name available in the namespace
                name = alias.asname if alias.asname else alias.name
                if name not in self.name_to_imports:
                    self.name_to_imports[name] = []
                self.name_to_imports[name].append((node, idx, alias))

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                # The name available in the namespace
                name = alias.asname if alias.asname else alias.name
                if name not in self.name_to_imports:
                    self.name_to_imports[name] = []
                self.name_to_imports[name].append((node, idx, alias))

    def _mark_duplicates_for_removal(self):
        """Mark all but the last import of each name for removal"""
        for _, imports in self.name_to_imports.items():
            if len(imports) <= 1:
                continue

            # Sort by index to find the last one
            imports.sort(key=lambda x: x[1])

            # Mark all but the last for removal
            for node, _, alias in imports[:-1]:
                if len(node.names) == 1:
                    self.nodes_to_remove.add(node)
                else:
                    if alias in node.names:
                        node.names.remove(alias)
                    if not node.names:
                        self.nodes_to_remove.add(node)
