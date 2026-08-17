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
    ``_CollectNames`` gathers the leading name of any dotted expression.  For
    ``numpy.ndarray`` it collects ``numpy``.  If ``numpy`` is imported, the
    whole expression is kept.  This is intentional: we can't validate that
    ``numpy.ndarray`` exists without importing the package, and removing half
    an attribute chain would produce invalid stubs.

Relationship to ``strip_artifacts``:
    Both modules rewrite expressions that reference names the module doesn't
    define, and both use a "root name of a dotted chain" resolvability check.
    They are not redundant, though - each covers a case the other doesn't:

    * This pass is general-purpose and user-configurable
      (``StubgenPyxConfig.trim_not_defined``). It catches *any* name that was
      never defined to begin with - most commonly a reference to a dependency
      that isn't available in the stub's environment. It covers default
      values as well as annotations, and it replaces with ``...`` to signal
      "unresolvable", since a real ``Any`` import may not be wanted for a
      pass a user can turn off.
    * ``strip_artifacts`` is unconditional and narrower: it only repairs
      annotations/decorators that reference a name *it just deleted itself*
      (a Cython-only import, a runtime-constant assignment), and does so
      with ``Any`` because it also manages the corresponding
      ``from typing import Any`` so the stub still parses regardless of user
      config.

    This pass runs first in the pipeline, so in practice it already collapses
    most dangling references to ``...`` before ``strip_artifacts`` sees them;
    ``strip_artifacts``'s own rewrite mainly matters when this pass is
    disabled, or for names it deletes itself later in the same pipeline pass.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from .utils import PUBLIC_BUILTIN_NAMES, dotted_name

logger = logging.getLogger(__name__)

# Built-in names that should never be trimmed.
_BUILTIN_NAMES = set(PUBLIC_BUILTIN_NAMES)

_BUILTIN_NAMES.update(
    {
        "__name__",
        "__doc__",
        "__file__",
        "__package__",
        "__loader__",
        "__spec__",
    }  # Names that are usually defined but not in builtins
)


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
    remover = _NotDefinedRemover(definitions, _contains_star_import(tree))
    tree = remover.visit(tree)

    for name in remover.replaced:
        logger.warning("Replaced undefined name %r with '...'", name)
    return tree


class _FoundStarImport(Exception):
    """Exception raised when a star import is encountered."""


class _SearchStarImports(ast.NodeVisitor):
    """Search for star imports, exiting early if found."""

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.names[0].name == "*":
            raise _FoundStarImport


def _contains_star_import(tree: ast.AST) -> bool:
    try:
        _SearchStarImports().visit(tree)
        return False
    except _FoundStarImport:
        return True


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
            if alias.asname:
                self.defined_names.add(alias.asname)
            else:
                self.defined_names.add(alias.name.split(".", 1)[0])

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

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Collect the leading name of a dotted expression."""
        root = dotted_name(node).split(".")[0]
        if root:
            self.names.add(root)

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
    contains_star_import: bool = False
    replaced: list[str] = field(default_factory=list, init=False)

    def _should_remove(self, used_names: set[str]) -> bool:
        """Check if any used names are undefined."""
        return not used_names.issubset(self.defined_names)

    def _replace_if_undefined(
        self, node: ast.expr, annotation: bool = False, type_alias: bool = False
    ) -> ast.expr:
        if annotation and self.contains_star_import:
            # Do not replace names in type annotations if a star import is present
            return node
        used_names: set[str] = set()
        _CollectNames(used_names).visit(node)
        undefined = used_names - self.defined_names
        if undefined:
            for name in sorted(undefined):
                self.replaced.append(name)
            if not type_alias:
                return ast.Constant(...)
            return ast.Name("Any", ast.Load())
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Process assignment values."""
        node.value = self._replace_if_undefined(node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Process annotated assignment annotation and value."""
        node.annotation = self._replace_if_undefined(node.annotation, annotation=True)
        if node.value is not None:
            type_alias_override = False
            if (
                isinstance(node.annotation, ast.Name)
                and node.annotation.id == "TypeAlias"
            ):
                type_alias_override = True  # Treat type aliases as annotations
            node.value = self._replace_if_undefined(
                node.value,
                annotation=type_alias_override,
                type_alias=type_alias_override,
            )
        return node

    def visit_arguments(self, node: ast.arguments) -> ast.arguments:
        """Process function argument annotations and defaults."""
        for arg in node.args + node.posonlyargs + node.kwonlyargs:
            if arg.annotation:
                arg.annotation = self._replace_if_undefined(
                    arg.annotation, annotation=True
                )

        node.defaults = [
            self._replace_if_undefined(default) for default in node.defaults
        ]

        node.kw_defaults = [
            self._replace_if_undefined(default) if default is not None else None
            for default in node.kw_defaults
        ]

        if node.vararg and node.vararg.annotation:
            node.vararg.annotation = self._replace_if_undefined(
                node.vararg.annotation, annotation=True
            )
        if node.kwarg and node.kwarg.annotation:
            node.kwarg.annotation = self._replace_if_undefined(
                node.kwarg.annotation, annotation=True
            )

        return node

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.AST:
        """Process function return type annotation."""
        if node.returns is not None:
            node.returns = self._replace_if_undefined(node.returns, annotation=True)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Process function definition."""
        return self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Process async function definition."""
        return self._process_function(node)
