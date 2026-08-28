"""Removes undefined names from Python .pyi files."""

from __future__ import annotations

import ast
import logging
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Literal, Union

from .utils import PUBLIC_BUILTIN_NAMES, dotted_name

_logger = logging.getLogger(__name__)

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

Scope = dict[str, Union[Literal[True], dict]]
Scopes = dict[tuple[str, ...], Scope]


def trim_not_defined(tree: ast.AST) -> ast.AST:
    """Remove undefined names from an AST. If an annotation is undefined, it is replaced with ``_typeshed.Incomplete``.

    Args:
        tree: The AST to process.

    Returns:
        Transformed AST without undefined names.
    """
    definitions: Scopes = {(): {}}
    collector = _DefinedCollector(definitions)
    collector.visit(tree)
    remover = _NotDefinedRemover(definitions, _contains_star_import(tree))
    tree = remover.visit(tree)

    for name in remover.replaced:
        _logger.debug("Trimmed undefined name %r", name)
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
    """Collect names defined in each module or class scope."""

    scopes: Scopes
    scope_key: tuple[str, ...] = ()

    @property
    def defined_names(self) -> Scope:
        return self.scopes[self.scope_key]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Collect function name; do NOT descend into the body."""
        self.defined_names[node.name] = True

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Collect async function name; do NOT descend into the body."""
        self.defined_names[node.name] = True

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Collect the class name and names in its nested scope."""
        class_scope: Scope = {}
        self.defined_names[node.name] = class_scope
        class_key = self.scope_key + (node.name,)
        self.scopes[class_key] = class_scope
        collector = _DefinedCollector(self.scopes, class_key)
        for child in node.body:
            collector.visit(child)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Collect annotated assignment names."""
        if isinstance(node.target, ast.Name):
            self.defined_names[node.target.id] = True
        # Do not call generic_visit: no nested scope to descend

    def visit_Assign(self, node: ast.Assign) -> None:
        """Collect assignment targets."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined_names[target.id] = True

    def visit_Import(self, node: ast.Import) -> None:
        """Collect import names."""
        for alias in node.names:
            if alias.asname:
                self.defined_names[alias.asname] = True
            else:
                self.defined_names[alias.name.split(".", 1)[0]] = True

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Collect from-import names."""
        for alias in node.names:
            self.defined_names[alias.asname if alias.asname else alias.name] = True


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


def _walk_names(root: ast.AST) -> Generator[str, None, None]:
    for node in ast.walk(root):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield dotted_name(node).split(".")[0]


def _collect_targets(root: ast.AST) -> set[str]:
    output = set()
    for node in ast.walk(root):
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            for gen in node.generators:
                output.update(_walk_names(gen.target))
        elif isinstance(node, ast.Lambda):
            output.update(arg.arg for arg in node.args.args)
    return output


def _typeshed_incomplete() -> ast.Attribute:
    return ast.Attribute(
        value=ast.Name(id="_typeshed", ctx=ast.Load()),
        attr="Incomplete",
        ctx=ast.Load(),
    )


@dataclass
class _NotDefinedRemover(ast.NodeTransformer):
    """Replace undefined name references with Ellipsis.

    Processes type annotations, default values, and return type annotations,
    replacing any expression containing undefined names with ``...``.

    Attributes:
        defined_names: Names defined in the module (not including builtins).
        replaced: Names that were replaced with Ellipsis.
    """

    scopes: Scopes
    contains_star_import: bool = False
    replaced: list[str] = field(default_factory=list, init=False)
    scope_key: tuple[str, ...] = field(default=(), init=False)

    def _defined_names(self, scope_key: tuple[str, ...]) -> set[str]:
        defined = set(_BUILTIN_NAMES)
        for end in range(len(scope_key) + 1):
            defined.update(self.scopes[scope_key[:end]])
        return defined

    def _check_expr_undefined(
        self,
        node: ast.expr,
        extra_defines: set[str] | None = None,
        scope_key: tuple[str, ...] | None = None,
    ) -> bool:
        used_names: set[str] = set()
        _CollectNames(used_names).visit(node)

        key = self.scope_key if scope_key is None else scope_key
        undefined = used_names - (self._defined_names(key) | (extra_defines or set()))
        for name in sorted(undefined):
            self.replaced.append(name)

        return bool(undefined)

    def _replace_value_if_undefined(self, node: ast.expr) -> ast.expr:
        """RHS of an assignment (this excludes TypeAlias assignments, which are treated as annotations)"""
        if self._check_expr_undefined(node, _collect_targets(node)):
            return ast.Constant(...)
        return node

    def _replace_annotation_if_undefined(self, node: ast.expr) -> ast.expr:
        if self.contains_star_import:
            return node
        if self._check_expr_undefined(node):
            return _typeshed_incomplete()
        return node

    def _remove_decorators_if_undefined(
        self, decorator_list: list[ast.expr]
    ) -> list[ast.expr]:
        output = []
        for dec in decorator_list:
            if self._check_expr_undefined(dec):
                continue
            output.append(dec)
        return output

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Process assignment values."""
        node.value = self._replace_value_if_undefined(node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Process annotated assignment annotation and value."""
        node.annotation = self._replace_annotation_if_undefined(node.annotation)
        if node.value is not None:
            annotation = dotted_name(node.annotation)
            if annotation == "typing_extensions.TypeAlias" or annotation == "TypeAlias":
                node.value = self._replace_annotation_if_undefined(node.value)
            else:
                node.value = self._replace_value_if_undefined(node.value)
        return node

    def visit_arguments(self, node: ast.arguments) -> ast.arguments:
        """Process function argument annotations and defaults."""
        for arg in node.args + node.posonlyargs + node.kwonlyargs:
            if arg.annotation:
                arg.annotation = self._replace_annotation_if_undefined(arg.annotation)

        node.defaults = [
            self._replace_value_if_undefined(default) for default in node.defaults
        ]

        node.kw_defaults = [
            self._replace_value_if_undefined(default) if default is not None else None
            for default in node.kw_defaults
        ]

        if node.vararg and node.vararg.annotation:
            node.vararg.annotation = self._replace_annotation_if_undefined(
                node.vararg.annotation
            )
        if node.kwarg and node.kwarg.annotation:
            node.kwarg.annotation = self._replace_annotation_if_undefined(
                node.kwarg.annotation
            )

        return node

    def _process_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.AST:
        """Process function return type annotation."""
        if node.returns is not None:
            node.returns = self._replace_annotation_if_undefined(node.returns)
        if node.decorator_list:
            node.decorator_list = self._remove_decorators_if_undefined(
                node.decorator_list
            )
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Process function definition."""
        return self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Process async function definition."""
        return self._process_function(node)

    def visit_ClassDef(self, node):
        """Process class decorators."""
        if node.decorator_list:
            node.decorator_list = self._remove_decorators_if_undefined(
                node.decorator_list
            )
        previous_scope = self.scope_key
        self.scope_key = previous_scope + (node.name,)
        out = self.generic_visit(node)
        self.scope_key = previous_scope
        return out
