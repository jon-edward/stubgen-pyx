"""Type parsing helpers for Cython AST nodes."""

from __future__ import annotations

from Cython.Compiler import Nodes, ExprNodes

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
