"""
Normalizes the names in a Python .pyi AST.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import itertools


def normalize_names(tree: ast.AST) -> ast.AST:
    """Normalizes the names in a Python .pyi AST, returning the normalized tree and the collected names of the script."""
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
    **{int_type: "int" for int_type in _CYTHON_INTS},
    **{float_type: "float" for float_type in _CYTHON_FLOATS},
    **{complex_type: "complex" for complex_type in _CYTHON_COMPLEXES},
}


@dataclass
class _NameNormalizer(ast.NodeTransformer):
    """Visits and normalizes the names in a Python .pyi AST, replacing Cython names with Python names."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        name = _CYTHON_TRANSLATIONS.get(node.id, node.id)
        return ast.Name(id=name, ctx=node.ctx)
