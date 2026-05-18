"""Trim undefined names from type annotations and default values.

Replaces references to names that are not builtin and not defined in the module
with `...` (Ellipsis). This is useful for stub files where external dependencies
should not be imported.

Examples:
    >>> code = 'def foo(x: UndefinedType = UNDEFINED_VALUE) -> int: pass'
    >>> tree = ast.parse(code)
    >>> trimmed = trim_not_defined(tree)
    >>> ast.unparse(trimmed)
    'def foo(x: ... = ...) -> int: pass'
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
from typing import Any


_BUILTIN_NAMES = {
    name for name in dir(builtins) if not name.startswith("_")
}  # Built-in names that should never be trimmed


def trim_not_defined(tree: ast.AST) -> ast.AST:
    """Remove undefined names from annotations and defaults in an AST.

    Scans the AST to collect all names defined via imports, assignments, function
    definitions, and class definitions, then replaces any undefined name references
    in type annotations, default values, and return type annotations with `...`.

    Args:
        tree: The AST module to process.

    Returns:
        Transformed AST with undefined names replaced by Ellipsis.
    """
    definitions = set()
    collector = _DefinedCollector(definitions)
    collector.visit(tree)
    definitions = definitions.union(_BUILTIN_NAMES)
    tree = _NotDefinedRemover(definitions).visit(tree)
    return tree


@dataclass
class _DefinedCollector(ast.NodeVisitor):
    """Collects all names defined in an AST via imports, assignments, and definitions."""

    defined_names: set[str]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Collect function name."""
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Collect async function name."""
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        """Collect class name."""
        self.defined_names.add(node.name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """Collect annotated assignment names."""
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Collect assignment targets."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_names.add(target.id)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> Any:
        """Collect import names."""
        for alias in node.names:
            self.defined_names.add(alias.asname if alias.asname else alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        """Collect from-import names."""
        for alias in node.names:
            self.defined_names.add(alias.asname if alias.asname else alias.name)
        self.generic_visit(node)


@dataclass
class _CollectNames(ast.NodeVisitor):
    """Collect all Name identifiers from an AST subtree."""

    names: set[str]

    def visit_Name(self, node: ast.Name) -> None:
        """Collect name identifier."""
        self.names.add(node.id)


@dataclass
class _NotDefinedRemover(ast.NodeTransformer):
    """Replace undefined name references with Ellipsis.

    Processes type annotations, default values, and return type annotations,
    replacing any expression containing undefined names with `...`.
    """

    defined_names: set[str]

    def _should_remove(self, used_names: set[str]) -> bool:
        """Check if any used names are undefined."""
        return not used_names.issubset(self.defined_names)

    def _replace_if_undefined(self, node: ast.expr) -> ast.expr:
        """Replace node with Ellipsis if it contains undefined names."""
        if node is None:
            return None
        used_names = set()
        _CollectNames(used_names).visit(node)
        if self._should_remove(used_names):
            return ast.Constant(...)
        return node

    def visit_Assign(self, node: ast.Assign) -> Any:
        """Process assignment values."""
        node.value = self._replace_if_undefined(node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        """Process annotated assignment annotation and value."""
        node.annotation = self._replace_if_undefined(node.annotation)
        if node.value is not None:
            node.value = self._replace_if_undefined(node.value)
        return node

    def visit_arguments(self, node: ast.arguments) -> Any:
        """Process function argument annotations and defaults."""
        # Process positional and positional-only argument annotations
        for arg in node.args + node.posonlyargs + node.kwonlyargs:
            if arg.annotation:
                arg.annotation = self._replace_if_undefined(arg.annotation)

        # Process positional and positional-only defaults
        node.defaults = [
            self._replace_if_undefined(default) for default in node.defaults
        ]

        # Process keyword-only defaults
        node.kw_defaults = [
            self._replace_if_undefined(default) if default is not None else None
            for default in node.kw_defaults
        ]

        # Process *args and **kwargs annotations
        if node.vararg and node.vararg.annotation:
            node.vararg.annotation = self._replace_if_undefined(node.vararg.annotation)
        if node.kwarg and node.kwarg.annotation:
            node.kwarg.annotation = self._replace_if_undefined(node.kwarg.annotation)

        return node

    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Any:
        """Process function return type annotation."""
        if node.returns is not None:
            node.returns = self._replace_if_undefined(node.returns)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """Process function definition."""
        return self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """Process async function definition."""
        return self._process_function(node)
