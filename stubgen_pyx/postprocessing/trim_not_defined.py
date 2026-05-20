"""Trim undefined names from type annotations and default values.

Replaces references to names that are not builtin and not defined in the module
with ``...`` (Ellipsis). This is useful for stub files where external dependencies
should not be imported.

Examples:
    >>> code = 'def foo(x: UndefinedType = UNDEFINED_VALUE) -> int: pass'
    >>> tree = ast.parse(code)
    >>> trimmed = trim_not_defined(tree)
    >>> ast.unparse(trimmed)
    'def foo(x: ... = ...) -> int: pass'

Note on attribute annotations (e.g. ``numpy.ndarray``):
    ``_CollectNames`` only gathers ``ast.Name`` nodes (i.e. the root name of
    any dotted expression).  For ``numpy.ndarray`` it collects ``numpy``.  If
    ``numpy`` is imported, the whole expression is kept.  This is intentional:
    we can't validate that ``numpy.ndarray`` exists without importing the
    package, and removing half an attribute chain would produce invalid stubs.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
import logging


logger = logging.getLogger(__name__)

_BUILTIN_NAMES = {
    name for name in dir(builtins) if not name.startswith("_")
}  # Built-in names that should never be trimmed


def trim_not_defined(tree: ast.AST) -> ast.AST:
    """Remove undefined names from annotations and defaults in an AST.

    Scans the module-level AST (without descending into nested function or
    class bodies for name collection, so scope leakage is avoided) to collect
    all names defined via imports, assignments, function definitions, and class
    definitions, then replaces any undefined name references in type
    annotations, default values, and return type annotations with ``...``.

    Warns if any undefined names are replaced.

    Args:
        tree: The AST module to process.

    Returns:
        Transformed AST with undefined names replaced by Ellipsis.
    """
    definitions: set[str] = set()
    collector = _DefinedCollector(definitions)
    collector.visit(tree)
    definitions = definitions | _BUILTIN_NAMES
    remover = _NotDefinedRemover(definitions)
    tree = remover.visit(tree)

    for name in remover.replaced:
        logger.warning("Replaced undefined name %r with '...'", name)
    return tree


@dataclass
class _DefinedCollector(ast.NodeVisitor):
    """Collect module-level defined names without leaking nested scopes.

    Visits the top-level body only.  Function and class bodies are not
    descended into, so locally-scoped names don't pollute the module-level
    definition set.
    """

    defined_names: set[str]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function name; do NOT descend into the body."""
        self.defined_names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async function name; do NOT descend into the body."""
        self.defined_names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Collect class name; do NOT descend into the body."""
        self.defined_names.add(node.name)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Collect annotated assignment names."""
        if isinstance(node.target, ast.Name):
            self.defined_names.add(node.target.id)
        # Do not call generic_visit: no nested scope to descend

    def visit_Assign(self, node: ast.Assign) -> None:
        """Collect assignment targets."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_names.add(target.id)

    def visit_Import(self, node: ast.Import) -> None:
        """Collect import names."""
        for alias in node.names:
            self.defined_names.add(alias.asname if alias.asname else alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Collect from-import names."""
        for alias in node.names:
            self.defined_names.add(alias.asname if alias.asname else alias.name)


@dataclass
class _CollectNames(ast.NodeVisitor):
    """Collect all Name identifiers from an AST subtree.

    Only ``ast.Name`` nodes are collected.  For attribute chains such as
    ``numpy.ndarray``, only the root name (``numpy``) is gathered.  See the
    module docstring for the rationale.
    """

    names: set[str]

    def visit_Name(self, node: ast.Name) -> None:
        """Collect name identifier."""
        self.names.add(node.id)


@dataclass
class _NotDefinedRemover(ast.NodeTransformer):
    """Replace undefined name references with Ellipsis.

    Processes type annotations, default values, and return type annotations,
    replacing any expression containing undefined names with ``...``.

    Attributes:
        defined_names: Names defined in the module (not including builtins).
        replaced: Names that were replaced with Ellipsis.
    """

    defined_names: set[str]
    replaced: list[str] = field(default_factory=list, init=False)

    def _should_remove(self, used_names: set[str]) -> bool:
        """Check if any used names are undefined."""
        return not used_names.issubset(self.defined_names)

    def _replace_if_undefined(self, node: ast.expr) -> ast.expr:
        if node is None:
            return None
        used_names: set[str] = set()
        _CollectNames(used_names).visit(node)
        undefined = used_names - self.defined_names
        if undefined:
            for name in sorted(undefined):
                self.replaced.append(name)
            return ast.Constant(...)
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Process assignment values."""
        node.value = self._replace_if_undefined(node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Process annotated assignment annotation and value."""
        node.annotation = self._replace_if_undefined(node.annotation)
        if node.value is not None:
            node.value = self._replace_if_undefined(node.value)
        return node

    def visit_arguments(self, node: ast.arguments) -> ast.arguments:
        """Process function argument annotations and defaults."""
        for arg in node.args + node.posonlyargs + node.kwonlyargs:
            if arg.annotation:
                arg.annotation = self._replace_if_undefined(arg.annotation)

        node.defaults = [
            self._replace_if_undefined(default) for default in node.defaults
        ]

        node.kw_defaults = [
            self._replace_if_undefined(default) if default is not None else None
            for default in node.kw_defaults
        ]

        if node.vararg and node.vararg.annotation:
            node.vararg.annotation = self._replace_if_undefined(node.vararg.annotation)
        if node.kwarg and node.kwarg.annotation:
            node.kwarg.annotation = self._replace_if_undefined(node.kwarg.annotation)

        return node

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.AST:
        """Process function return type annotation."""
        if node.returns is not None:
            node.returns = self._replace_if_undefined(node.returns)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Process function definition."""
        return self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Process async function definition."""
        return self._process_function(node)
