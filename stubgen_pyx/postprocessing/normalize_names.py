"""Normalizes Cython type names to Python equivalents in .pyi files."""

from __future__ import annotations

import ast
from dataclasses import dataclass


def normalize_names(tree: ast.AST) -> ast.AST:
    """Replace Cython type names with their Python equivalents in an AST."""
    return _NameNormalizer().visit(tree)


_CYTHON_INTS: tuple[str, ...] = (
    "char",
    "short",
    "Py_UNICODE",
    "Py_UCS4",
    "long",
    "longlong",
    "Py_hash_t",
    "Py_ssize_t",
    "size_t",
    "ssize_t",
    "ptrdiff_t",
    "int64_t",
    "int32_t",
)

_CYTHON_FLOATS: tuple[str, ...] = (
    "double",
    "longdouble",
)

_CYTHON_COMPLEXES: tuple[str, ...] = (
    "longdoublecomplex",
    "doublecomplex",
    "floatcomplex",
)

_CYTHON_TRANSLATIONS: dict[str, str] = {
    "bint": "bool",
    "unicode": "str",
    "void": "None",
}

for int_type in _CYTHON_INTS:
    _CYTHON_TRANSLATIONS[int_type] = "int"
for float_type in _CYTHON_FLOATS:
    _CYTHON_TRANSLATIONS[float_type] = "float"
for complex_type in _CYTHON_COMPLEXES:
    _CYTHON_TRANSLATIONS[complex_type] = "complex"


@dataclass
class _NameNormalizer(ast.NodeTransformer):
    """Transform Cython type names to Python equivalents."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        name = _CYTHON_TRANSLATIONS.get(node.id, node.id)
        return ast.Name(id=name, ctx=node.ctx)
