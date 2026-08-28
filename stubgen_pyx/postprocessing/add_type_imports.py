"""
Add required type imports to a Python .pyi file.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from .collect_names import collect_names
from .utils import dotted_name

TYPE_IMPORTS = (
    "_typeshed.Incomplete",
    "typing.Any",
    "typing.Callable",
    "typing.TypedDict",
    "typing.TypeVar",
    "typing_extensions.TypeAlias",  # Use backport for Python < 3.10
    "enum.IntEnum",
    "numpy.typing.NDArray",
    "numpy",
)
# Qualified names for imported types that
# might be needed by stubs.


def add_type_imports(node: ast.AST) -> ast.AST:
    """Add imports for supported qualified type names to a stub AST."""
    if not isinstance(node, ast.Module):
        raise TypeError("node must be an ast.Module")

    used_names = collect_names(node, add_declared=True)
    qualified_names = _collect_qualified_names(node)
    existing_imports = _collect_imports(node)
    added_imports, renamed = _resolve_type_imports(
        qualified_names, existing_imports, used_names
    )

    _prepend_imports(node, added_imports)
    _AttributeRenamer(renamed).visit(node)
    return node


def _collect_qualified_names(node: ast.AST) -> set[str]:
    """Return dotted attribute names referenced by ``node``."""
    collector = _QualifiedNameCollector()
    collector.visit(node)
    return collector.names


def _collect_imports(node: ast.AST) -> set[_Import]:
    """Return all regular imports declared in ``node``."""
    visitor = _ImportVisitor()
    visitor.visit(node)
    return visitor.imports


def _resolve_type_imports(
    qualified_names: set[str],
    existing_imports: set[_Import],
    used_names: set[str],
) -> tuple[set[_Import], dict[str, str]]:
    """Resolve required imports and qualified-name replacements."""
    added_imports: set[_Import] = set()
    renamed: dict[str, str] = {}

    for qualified_name in TYPE_IMPORTS:
        if not _is_used(qualified_name, qualified_names):
            continue
        import_to_add, replacement = _resolve_type_import(
            qualified_name, existing_imports, used_names
        )
        if import_to_add is not None:
            added_imports.add(import_to_add)
        if replacement is not None:
            renamed[qualified_name] = replacement

    return added_imports, renamed


def _is_used(qualified_name: str, names: set[str]) -> bool:
    """Return whether a name or one of its qualified descendants is used."""
    return qualified_name in names or any(
        name.startswith(f"{qualified_name}.") for name in names
    )


def _resolve_type_import(
    qualified_name: str,
    existing_imports: set[_Import],
    used_names: set[str],
) -> tuple[_Import | None, str | None]:
    """Return an import to add and replacement for one qualified name."""
    if "." not in qualified_name:
        existing = _find_import(existing_imports, qualified_name, None)
        if existing and existing.asname:
            return None, existing.asname
        return (None, None) if existing else (_Import(qualified_name, None, None), None)

    root, import_name = qualified_name.rsplit(".", 1)
    existing = _find_import(existing_imports, root, import_name)
    if existing is not None:
        return None, existing.asname or import_name

    module_import = _find_longest_module_import(qualified_name, existing_imports)
    if module_import is not None:
        if module_import.asname:
            suffix = qualified_name[len(module_import.module) + 1 :]
            return None, f"{module_import.asname}.{suffix}"
        return None, None

    if import_name not in used_names:
        return _Import(root, import_name, None), import_name
    return _Import(root, None, None), None


def _find_import(
    imports: set[_Import], module: str, name: str | None
) -> _Import | None:
    """Find an import matching a module and optional imported name."""
    return next(
        (
            candidate
            for candidate in imports
            if candidate.module == module and candidate.name == name
        ),
        None,
    )


def _find_longest_module_import(
    qualified_name: str, imports: set[_Import]
) -> _Import | None:
    """Find the most specific plain module import covering ``qualified_name``."""
    module_imports = (
        candidate
        for candidate in imports
        if candidate.name is None
        and (
            qualified_name == candidate.module
            or qualified_name.startswith(f"{candidate.module}.")
        )
    )
    return max(
        module_imports, key=lambda candidate: len(candidate.module), default=None
    )


def _first_idx_after_docstring(node: ast.Module) -> int:
    if not node.body:
        return 0
    if isinstance(node.body[0], ast.Expr) and isinstance(
        node.body[0].value, ast.Constant
    ):
        return 1
    return 0


def _prepend_imports(node: ast.Module, imports: set[_Import]) -> None:
    """Prepend newly required imports to a module."""
    if not imports:
        return
    insert_idx = _first_idx_after_docstring(node)
    node.body[insert_idx:insert_idx] = [
        _to_ast_import(import_) for import_ in sorted(imports, key=_import_sort_key)
    ]


def _import_sort_key(import_: _Import) -> tuple[str, bool, str]:
    """Sort module imports before from-imports from the same module."""
    return import_.module, import_.name is not None, import_.name or ""


def _to_ast_import(import_: _Import) -> ast.Import | ast.ImportFrom:
    """Convert an internal import description to an AST import node."""
    if import_.name is None:
        return ast.Import(names=[ast.alias(name=import_.module)])
    return ast.ImportFrom(
        module=import_.module,
        names=[ast.alias(name=import_.name)],
        level=0,
    )


@dataclass(frozen=True)
class _Import:
    module: str
    name: str | None
    asname: str | None


@dataclass
class _ImportVisitor(ast.NodeVisitor):
    """Collect imports from a module without traversing imported names."""

    imports: set[_Import] = field(default_factory=set)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imports.add(_Import(alias.name, None, alias.asname))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if node.module is None:
                continue
            self.imports.add(_Import(node.module, alias.name, alias.asname))


@dataclass
class _QualifiedNameCollector(ast.NodeVisitor):
    """Collect dotted attribute expressions from an AST."""

    names: set[str] = field(default_factory=set)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.names.add(dotted_name(node))
        self.generic_visit(node)


@dataclass
class _AttributeRenamer(ast.NodeTransformer):
    """Replace selected dotted attribute expressions with shorter names."""

    replace: dict[str, str]

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        name = dotted_name(node)
        if name in self.replace:
            replacement = self.replace[name]
            replacement_parts = replacement.split(".")
            replacement_node: ast.expr = ast.Name(id=replacement_parts[0], ctx=node.ctx)
            for part in replacement_parts[1:]:
                replacement_node = ast.Attribute(
                    value=replacement_node,
                    attr=part,
                    ctx=node.ctx,
                )
            return ast.copy_location(replacement_node, node)
        return node
