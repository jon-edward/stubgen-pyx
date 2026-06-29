"""Utility functions for converting Cython AST nodes."""

from __future__ import annotations

import logging
import textwrap

from Cython.Compiler import Nodes, ExprNodes
from Cython.CodeWriter import ExpressionWriter

_CYTHON_TO_NUMPY_SCALAR: dict[str, str] = {
    "bint": "bool_",
    "bool": "bool_",
    "char": "byte",
    "signed char": "int8",
    "short": "short",
    "short int": "short",
    "int": "intc",
    "long": "int_",
    "long int": "int_",
    "long long": "longlong",
    "long long int": "longlong",
    "unsigned char": "ubyte",
    "unsigned short": "ushort",
    "unsigned short int": "ushort",
    "unsigned int": "uintc",
    "unsigned long": "uint",
    "unsigned long int": "uint",
    "unsigned long long": "ulonglong",
    "unsigned long long int": "ulonglong",
    "int8_t": "int8",
    "int16_t": "int16",
    "int32_t": "int32",
    "int64_t": "int64",
    "uint8_t": "uint8",
    "uint16_t": "uint16",
    "uint32_t": "uint32",
    "uint64_t": "uint64",
    "Py_ssize_t": "intp",
    "size_t": "uintp",
    "Py_intptr_t": "intp",
    "float": "single",
    "double": "double",
    "long double": "longdouble",
    "float complex": "complex64",
    "double complex": "complex128",
}


def extract_type_from_base_type(node, is_ptr: bool = False) -> str | None:
    """Extract a type annotation string from a base_type node.

    Handles plain named types, pointer types (``char *`` -> ``bytes``),
    tuple types, C++ templated types, fixed-size C arrays, and typed
    memoryviews.
    """
    try:
        base_type = node.base_type
    except AttributeError:
        return None

    # CArgDeclNode carries a single .declarator; check it for pointer-ness.
    if not is_ptr:
        is_ptr = isinstance(getattr(node, "declarator", None), Nodes.CPtrDeclaratorNode)

    if isinstance(base_type, Nodes.CTupleBaseTypeNode):
        return _extract_tuple_type(base_type)
    if isinstance(base_type, Nodes.TemplatedTypeNode):
        return _extract_templated_type(base_type)
    if isinstance(base_type, Nodes.MemoryViewSliceTypeNode):
        return _extract_memoryview_type(base_type)

    name: str | None = None
    if hasattr(base_type, "name") and base_type.name is not None:
        name = ".".join(base_type.module_path + [base_type.name])
    if hasattr(base_type, "base_type_node") and base_type.base_type_node is not None:
        name = ".".join(
            base_type.base_type_node.module_path + [base_type.base_type_node.name]
        )

    if is_ptr and name == "char":
        return "bytes"

    return name


def _extract_tuple_type(node: Nodes.CTupleBaseTypeNode) -> str:
    """Unparse a C tuple base-type node as ``tuple[A, B, ...]``."""
    parts = [extract_type_from_base_type(c) or "object" for c in node.components]
    return f"tuple[{', '.join(parts)}]"


def _extract_templated_type(node: Nodes.TemplatedTypeNode) -> str | None:
    """Unparse a ``TemplatedTypeNode`` as either a fixed-size C array or a
    C++ template instantiation.

    Fixed-size C arrays (``char[100]``, ``int[100][100]``) are detected by
    their positional args being integer literals and are delegated to
    ``_extract_array_type``.  Everything else is treated as a C++ template
    and rendered as ``Base[T1, T2, ...]``.  Returns ``None`` when the base
    type cannot be resolved.
    """
    positional_args = getattr(node, "positional_args", [])

    # Fixed-size C array: all positional args are integer literals.
    if positional_args and all(
        isinstance(a, ExprNodes.IntNode) for a in positional_args
    ):
        return _extract_array_type(node)

    # Template instantiation
    base_type_node = getattr(node, "base_type_node", None)
    if base_type_node is None:
        return None

    base = ".".join(base_type_node.module_path + [base_type_node.name])

    parts = [extract_type_from_base_type(a) or "object" for a in positional_args]
    # keywords intentionally omitted as they are not supported in Python
    return f"{base}[{', '.join(parts)}]" if parts else base


def _extract_array_type(node: Nodes.TemplatedTypeNode) -> str | None:
    """Recursively unwrap nested ``TemplatedTypeNode`` fixed-size C arrays.

    ``char[100]``      -> ``"bytes"``
    ``char[100][100]`` -> ``"list[bytes]"``
    ``int[100]``       -> ``"list[int]"``
    ``int[100][100]``  -> ``"list[list[int]]"``

    Returns ``None`` when the innermost base type cannot be resolved.
    """
    base_type_node = getattr(node, "base_type_node", None)
    if base_type_node is None:
        return None

    # Nested array: recurse to resolve the inner type first.
    if isinstance(base_type_node, Nodes.TemplatedTypeNode):
        inner = _extract_array_type(base_type_node)
        return f"list[{inner}]" if inner is not None else None

    # Innermost level: resolve the scalar name.
    try:
        name = ".".join(base_type_node.module_path + [base_type_node.name])
    except AttributeError:
        return None

    if not name:
        return None
    return "bytes" if name == "char" else f"list[{name}]"


def _extract_memoryview_type(node) -> str:
    """Unparse a typed memoryview node as ``numpy.typing.NDArray[dtype]``.

    Falls back to plain ``memoryview`` when the scalar type is not in the
    mapping (e.g. a user-defined struct or an unrecognised C type).
    """
    base = getattr(node, "base_type_node", None)
    if base is not None:
        name = getattr(base, "name", None)
        scalar = None if name is None else _CYTHON_TO_NUMPY_SCALAR.get(name)
        if scalar:
            return f"numpy.typing.NDArray[numpy.{scalar}]"
    return "memoryview"


def get_source(source: str, node: Nodes.Node) -> str:
    """Extract source code for a node, dedented and stripped.

    ``end_pos`` is often inaccurate in Cython's AST; the function falls back
    to the start position when it is missing.
    """
    lines = source.splitlines(keepends=True)
    end_pos = node.end_pos() or node.pos
    output = "".join(lines[i - 1] for i in range(node.pos[1], end_pos[1] + 1))
    return textwrap.dedent(output).rstrip()


def get_decorators(
    source: str,
    node: Nodes.DefNode
    | Nodes.CFuncDefNode
    | Nodes.CClassDefNode
    | Nodes.PyClassDefNode,
) -> list[str]:
    """Return decorator source strings for a function or class node."""
    if node.decorators:
        return [get_source(source, d) for d in node.decorators]
    return []


def get_bases(node: Nodes.CClassDefNode | Nodes.PyClassDefNode) -> list[str]:
    """Return base-class name strings from a class node."""
    if not hasattr(node, "bases") or not node.bases:
        return []
    return [
        base.name for base in node.bases.args if isinstance(base, ExprNodes.NameNode)
    ]


def get_metaclass(node: Nodes.PyClassDefNode | Nodes.CClassDefNode) -> str | None:
    """Return the metaclass name from a Python class node, if present."""
    if not isinstance(node, Nodes.PyClassDefNode):
        return None
    if node.metaclass and isinstance(node.metaclass, ExprNodes.NameNode):
        return node.metaclass.name
    return None


def _declarator_name(
    decl: Nodes.CNameDeclaratorNode
    | Nodes.CPtrDeclaratorNode
    | Nodes.CConstDeclaratorNode,
) -> str:
    """Recursively unwrap pointer/const/func declarators to reach the name."""
    if isinstance(
        decl,
        (
            Nodes.CPtrDeclaratorNode,
            Nodes.CConstDeclaratorNode,
            Nodes.CFuncDeclaratorNode,
        ),
    ):
        return _declarator_name(decl.base)
    return decl.name


def get_cdef_variables(node: Nodes.CVarDefNode) -> list[tuple[str, str | None]]:
    """Return ``(name, type)`` pairs for every declarator in a cdef statement.

    A single ``cdef public int x, y, z`` node can contain multiple declarators.
    Fixed-size array types (``char[N]``, ``int[N][M]``) are resolved via the
    base_type's ``TemplatedTypeNode``; pointer declarators on ``char`` emit
    ``"bytes"``; function-pointer declarators emit ``"Callable"``.

    .. todo::
        Properly extract return and argument types for function pointers.
    """
    accepted = (
        Nodes.CNameDeclaratorNode,
        Nodes.CPtrDeclaratorNode,
        Nodes.CConstDeclaratorNode,
        Nodes.CFuncDeclaratorNode,
    )
    declarators = []
    for decl in node.declarators:
        if isinstance(decl, accepted):
            declarators.append(decl)
        else:
            logging.warning("Unknown declarator type: %s", type(decl).__name__)

    results = []
    for d in declarators:
        name = _declarator_name(d)
        if isinstance(d, Nodes.CFuncDeclaratorNode):
            typ: str | None = "Callable"
        elif isinstance(d, Nodes.CPtrDeclaratorNode):
            typ = extract_type_from_base_type(node, is_ptr=True)
        else:
            typ = extract_type_from_base_type(node)
        results.append((name, typ))
    return results


def get_enum_names(node: Nodes.CEnumDefNode) -> list[str]:
    """Return member names from an enum definition node."""
    return [item.name for item in node.items]  # type: ignore


def docstring_to_string(docstring: str) -> str:
    """Wrap a raw docstring in triple-double-quotes, escaping embedded ones."""
    if not docstring:
        return '""" """'
    first_line, *rest = docstring.splitlines(keepends=True)
    rest_joined = textwrap.dedent("".join(rest))
    body = f"{first_line}{rest_joined}".replace('"""', r"\"\"\"")
    return f'"""{body}"""'


def unparse_expr(node: Nodes.Node | None) -> str | None:
    """Render an expression node to source code."""
    if node is None:
        return None

    expr_writer = ExpressionWriter(allow_unknown_nodes=True)
    expr_writer.visit(node)
    return expr_writer.result
