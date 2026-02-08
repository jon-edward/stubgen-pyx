"""Collects referenced names from a .pyi AST for import trimming."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import itertools


def collect_names(tree: ast.AST) -> set[str]:
    """Extract all names referenced in a .pyi file's AST."""
    collector = _NameCollector()
    collector.visit(tree)
    return collector.names


@dataclass
class _NameCollector(ast.NodeVisitor):
    """Visitor that collects all referenced names from type annotations and code."""

    names: set[str] = field(default_factory=set, init=False)

    def _try_parsed_visit(self, str_constant: str) -> None:
        """Parse and visit string annotations (PEP 563 forward references)."""
        try:
            subtree = ast.parse(str_constant)
        except SyntaxError:
            subtree = None
        if subtree is None:
            return
        self.visit(subtree)

    @staticmethod
    def _get_str_constant(node: ast.AST | None) -> str | None:
        """Extract string value from ast.Constant node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value

    def _visit_arguments(self, args: ast.arguments):
        """Collect names from function argument annotations."""
        extra_args = []
        if args.vararg:
            extra_args.append(args.vararg)
        if args.kwarg:
            extra_args.append(args.kwarg)

        all_args = itertools.chain(
            args.args, args.kwonlyargs, args.posonlyargs, extra_args
        )
        for arg in all_args:
            str_constant = self._get_str_constant(arg.annotation)
            if str_constant:
                self._try_parsed_visit(str_constant)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef):
        """Collect names from function signature."""
        self._visit_arguments(node.args)
        returns_constant = self._get_str_constant(node.returns)
        if returns_constant:
            self._try_parsed_visit(returns_constant)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._visit_function(node)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._visit_function(node)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        """Collect names from annotated assignments."""
        str_constant = self._get_str_constant(node.annotation)
        if str_constant:
            self._try_parsed_visit(str_constant)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Collect module names accessed via attribute chains (e.g., os.path)."""
        names = []
        attribute = node

        while isinstance(attribute, ast.Attribute):
            names.append(attribute.attr)
            attribute = attribute.value

        if isinstance(attribute, ast.Name):
            names.append(attribute.id)

        names.reverse()
        names.pop()

        for i in range(1, len(names) + 1):
            self.names.add(".".join(names[0:i]))

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Collect referenced identifiers."""
        self.names.add(node.id)
        return node
