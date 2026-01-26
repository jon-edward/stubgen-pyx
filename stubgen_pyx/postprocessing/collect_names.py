"""
Collects names from a Python .pyi AST.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import itertools


def collect_names(tree: ast.AST) -> set[str]:
    """Collects names from a Python .pyi AST."""
    collector = _NameCollector()
    collector.visit(tree)
    return collector.names


@dataclass
class _NameCollector(ast.NodeVisitor):
    names: set[str] = field(default_factory=set, init=False)

    def _try_parsed_visit(self, str_constant: str) -> None:
        """Tries to visit a grafted AST segment in the current parsing context. If this fails, do nothing"""
        try:
            subtree = ast.parse(str_constant)
        except SyntaxError:
            subtree = None
        if subtree is None:
            return
        self.visit(subtree)

    @staticmethod
    def _get_str_constant(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value

    def _visit_arguments(self, args: ast.arguments):
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
        str_constant = self._get_str_constant(node.annotation)
        if str_constant:
            self._try_parsed_visit(str_constant)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
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
            # Add all potentially used module names to access the
            # attribute
            self.names.add(".".join(names[0:i]))

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        self.names.add(node.id)
        return node
