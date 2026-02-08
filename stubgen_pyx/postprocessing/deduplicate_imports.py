"""Removes duplicate import statements, keeping only the last occurrence."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


def deduplicate_imports(node: ast.AST) -> ast.Module:
    """Remove duplicate imports from an AST, keeping the last occurrence."""
    return _DuplicateImportRemover().visit(node)


@dataclass
class _DuplicateImportRemover(ast.NodeTransformer):
    """Remove all but the last import of each name.

    In Python, when a name is imported multiple times, the last import wins.
    This removes earlier imports, keeping only the final one.
    """

    name_to_imports: dict[
        str, list[tuple[ast.Import | ast.ImportFrom, int, ast.alias]]
    ] = field(default_factory=dict)
    nodes_to_remove: set[ast.AST] = field(default_factory=set)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """First pass: collect imports. Second pass: remove duplicates."""
        for idx, stmt in enumerate(node.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                self._register_import(stmt, idx)

        self._mark_duplicates_for_removal()

        new_body = []
        for idx, stmt in enumerate(node.body):
            if stmt not in self.nodes_to_remove:
                new_body.append(self.generic_visit(stmt))

        node.body = new_body
        return node

    def _register_import(self, node: ast.Import | ast.ImportFrom, idx: int):
        """Track all imports and the names they provide."""
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                if name not in self.name_to_imports:
                    self.name_to_imports[name] = []
                self.name_to_imports[name].append((node, idx, alias))

        elif isinstance(node, ast.ImportFrom):
            if any(alias.name == "*" for alias in node.names):
                key = f"_star_import_:{node.module}"
                if key not in self.name_to_imports:
                    self.name_to_imports[key] = []
                self.name_to_imports[key].append((node, idx, node.names[0]))
            else:
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name not in self.name_to_imports:
                        self.name_to_imports[name] = []
                    self.name_to_imports[name].append((node, idx, alias))

    def _mark_duplicates_for_removal(self):
        """Mark all but the last import of each name for removal."""
        for _, imports in self.name_to_imports.items():
            if len(imports) <= 1:
                continue

            imports.sort(key=lambda x: x[1])

            for node, _, alias in imports[:-1]:
                if len(node.names) == 1:
                    self.nodes_to_remove.add(node)
                else:
                    node.names[:] = [a for a in node.names if a is not alias]
                    if not node.names:
                        self.nodes_to_remove.add(node)
