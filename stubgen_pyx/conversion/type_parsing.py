"""Type parsing helpers for Cython AST nodes."""

from __future__ import annotations

import logging

from Cython.Compiler import ExprNodes, Nodes

from ..logging_utils import with_debug_fallback
from .unparse import unparse_expr

_logger = logging.getLogger(__name__)

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

_CYTHON_BUILTIN_GENERIC_MAPPING: dict[str, str] = {
    "tuple": "tuple[typing.Any, ...]",
    "list": "list[typing.Any]",
    "dict": "dict[typing.Any, typing.Any]",
    "set": "set[typing.Any]",
}


def parameterize_builtin_generic(name: str | None) -> str | None:
    """Map bare Cython container names to Any-filled Python generics."""
    if name is None:
        return None
    return _CYTHON_BUILTIN_GENERIC_MAPPING.get(name, name)


def _declarator_name(
    decl: Nodes.CNameDeclaratorNode
    | Nodes.CPtrDeclaratorNode
    | Nodes.CConstDeclaratorNode,
) -> str | None:
    """Recursively unwrap pointer/const/func declarators to reach the name."""
    if isinstance(
        decl,
        (
            Nodes.CPtrDeclaratorNode,
            Nodes.CConstDeclaratorNode,
            Nodes.CFuncDeclaratorNode,
            Nodes.CArrayDeclaratorNode,
        ),
    ):
        return _declarator_name(decl.base)
    return getattr(decl, "name", None)


def _get_func_decl_type(
    decl: Nodes.CFuncDeclaratorNode, base_type: str | None
) -> str | None:
    if not isinstance(decl, Nodes.CFuncDeclaratorNode):
        return None

    args = []
    for arg_idx, arg in enumerate(decl.args):
        base_type_arg = extract_type_from_base_type(arg)

        typ_arg = _get_cdef_declarator_type(arg.declarator, base_type_arg)[1]
        typ_arg = with_debug_fallback(
            typ_arg,
            "_typeshed.Incomplete",
            lambda arg_idx_=arg_idx, decl_=decl: (
                f"Replaced argument {arg_idx_} type in function {decl_.name} with 'Incomplete'"
            ),
        )

        args.append(typ_arg)
    base_type = with_debug_fallback(
        base_type,
        "_typeshed.Incomplete",
        lambda: f"Replaced return type in function {decl.name} with 'Incomplete'",
    )
    return f"typing.Callable[[{', '.join(args)}], {base_type}]"


def _get_cdef_declarator_type(
    decl, base_type: str | None = None
) -> tuple[str | None, str | None]:
    name = _declarator_name(decl)
    typ = None
    if isinstance(decl, Nodes.CFuncDeclaratorNode):
        typ = _get_func_decl_type(decl, base_type)
        typ = with_debug_fallback(
            typ,
            "typing.Callable[..., _typeshed.Incomplete]",
            lambda: (
                f"Replaced function {name} with 'typing.Callable[..., _typeshed.Incomplete]'"
            ),
        )
    elif isinstance(decl, Nodes.CPtrDeclaratorNode):
        decl = decl.base
        _, typ = _get_cdef_declarator_type(decl, base_type)
    elif isinstance(decl, Nodes.CNameDeclaratorNode):
        pass
    return (name, typ or base_type)


def extract_name_and_type(node) -> tuple[str | None, str | None]:
    base_type = extract_type_from_base_type(node)
    name, typ = _get_cdef_declarator_type(node.declarator, base_type=base_type)
    return name, typ


def get_cdef_variables(node: Nodes.CVarDefNode) -> list[tuple[str, str | None]]:
    """Return ``(name, type)`` pairs for every declarator in a cdef statement.

    A single ``cdef public int x, y, z`` node can contain multiple declarators.
    Fixed-size array types (``char[N]``, ``int[N][M]``) are resolved via the
    base_type's ``TemplatedTypeNode``; pointer declarators on ``char`` emit
    ``"bytes"``; function-pointer declarators emit ``"Callable"``.
    """
    accepted = (
        Nodes.CNameDeclaratorNode,
        Nodes.CPtrDeclaratorNode,
        Nodes.CConstDeclaratorNode,
        Nodes.CFuncDeclaratorNode,
        Nodes.CArrayDeclaratorNode,
    )
    declarators = []
    for decl in node.declarators:
        if isinstance(decl, accepted):
            declarators.append(decl)
        else:
            _logger.debug("Unknown declarator type: %s", type(decl).__name__)

    is_ptr = (
        False
        if not declarators
        else isinstance(declarators[0], Nodes.CPtrDeclaratorNode)
    )
    base_type = extract_type_from_base_type(node, is_ptr=is_ptr)

    results = []
    for d in declarators:
        results.append(_get_cdef_declarator_type(d, base_type=base_type))
    return results


def get_enum_names(node: Nodes.CEnumDefNode) -> list[str]:
    """Return member names from an enum definition node."""
    return [item.name for item in node.items]  # type: ignore


def _type_from_base_type_name(base_type) -> str | None:
    name: str | None = None
    if hasattr(base_type, "name") and base_type.name is not None:
        module_path = getattr(base_type, "module_path", [])
        name = ".".join(module_path + [base_type.name])
    if hasattr(base_type, "base_type_node") and base_type.base_type_node is not None:
        module_path = getattr(base_type.base_type_node, "module_path", [])
        name = ".".join(
            base_type.base_type_node.module_path + [base_type.base_type_node.name]
        )
    return name


def extract_type_from_base_type(node, is_ptr: bool = False) -> str | None:
    """Extract a type annotation string from a base_type node.

    Handles plain named types, pointer types (``char *`` -> ``bytes``),
    tuple types, C++ templated types, fixed-size C arrays, and typed
    memoryviews.
    """
    try:
        base_type = node.base_type
        if isinstance(base_type, Nodes.CConstOrVolatileTypeNode):
            base_type = base_type.base_type
    except AttributeError:
        if isinstance(node, ExprNodes.ExprNode):
            return unparse_expr(node)
        _logger.debug("Unknown base type: %s", type(node).__name__)
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

    name = _type_from_base_type_name(base_type)

    if isinstance(node, Nodes.CVarDefNode) and name is None:
        # CVarDefNode may not have a named type, e.g. ``cdef public x``.
        # In this case, use ``typing.Any`` without debug message.
        return "typing.Any"

    if is_ptr and name == "char":
        return "bytes"

    if is_ptr and name == "void":
        return "typing.Any"

    return parameterize_builtin_generic(name)


def _extract_tuple_type(node: Nodes.CTupleBaseTypeNode) -> str:
    """Unparse a C tuple base-type node as ``tuple[A, B, ...]``."""

    def _extract_type(c, c_idx: int):
        typ = with_debug_fallback(
            extract_type_from_base_type(c),
            "object",
            lambda c_idx_=c_idx, c_=c: (
                f"Replaced tuple component {unparse_expr(c_)} at index {c_idx_} with 'object'"
            ),
        )
        return typ

    parts = [_extract_type(c, c_idx) for c_idx, c in enumerate(node.components)]
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

    def _extract_type(a, a_idx: int):
        typ = with_debug_fallback(
            extract_type_from_base_type(a),
            "_typeshed.Incomplete",
            lambda: (
                f"Replaced template argument of {base} at index {a_idx} with '_typeshed.Incomplete'"
            ),
        )
        return typ

    parts = [_extract_type(a, a_idx) for a_idx, a in enumerate(positional_args)]
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
