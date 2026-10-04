"""Type parsing helpers for Cython AST nodes."""

from __future__ import annotations

import logging

from Cython.Compiler import ExprNodes, Nodes
from Cython.Compiler import PyrexTypes as _PyrexTypes

from ..logging_utils import with_debug_fallback
from .pyrex_types import (
    CYTHON_TO_NUMPY_SCALAR,
    parameterize_builtin_generic,
    render_pyrex_type,
)
from .unparse import unparse_expr
from .utils import decode_or_pass

_logger = logging.getLogger(__name__)

# Cython 3.3 renamed the const/volatile declarator and base-type node
# classes (`CConstDeclaratorNode` -> `CQualifierDeclaratorNode`,
# `CConstOrVolatileTypeNode` -> `CQualifierTypeNode`, the latter now
# also covering `restrict`), keeping the same attribute shape
# (`.base`/`.base_type`, `.is_const`, `.is_volatile`). Resolved once
# here so the rest of this module can use one name regardless of which
# Cython version (>=3.0) is installed.
_ConstDeclaratorNode = (
    getattr(Nodes, "CQualifierDeclaratorNode", None) or Nodes.CConstDeclaratorNode
)
_ConstOrVolatileTypeNode = (
    getattr(Nodes, "CQualifierTypeNode", None) or Nodes.CConstOrVolatileTypeNode
)


def _extract_resolved_type(node) -> str | None:
    resolved_type = getattr(node, "type", None) or getattr(
        getattr(node, "entry", None), "type", None
    )
    expected = resolved_type is None or (
        resolved_type is _PyrexTypes.py_object_type
        or isinstance(node, (Nodes.CFuncDefNode, Nodes.DefNode))
    )
    if expected:
        if resolved_type is None:
            _logger.debug("Unknown base type: %s", type(node).__name__)
        return None
    rendered = render_pyrex_type(resolved_type)
    if rendered is not None:
        return rendered
    _logger.debug("Unknown base type: %s", type(node).__name__)
    return None


def _extract_structural_type(node, base_type, is_ptr: bool) -> str | None:
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
        return "typing.Any"
    if is_ptr and name == "char":
        return "bytes"
    if is_ptr and name == "void":
        return "typing.Any"
    return parameterize_builtin_generic(name)


def extract_type_from_base_type(node, is_ptr: bool = False) -> str | None:
    """Extract a type annotation string from a base_type node.

    Handles plain named types, pointer types (``char *`` -> ``bytes``),
    tuple types, C++ templated types, fixed-size C arrays, and typed
    memoryviews.

    Checks for a value stashed by `static_annotations.capture_static_types`
    first: some
    Python-generic-parameterized types (``list[int]``) lose their
    parameters once resolved to a real `PyrexTypes.Type` (there's no
    equivalent to `CppClassType.templates` for them), so a pre-pipeline
    snapshot of the structural extraction can be strictly more precise
    than either a live walk of the (possibly since-cleared) node or the
    `Entry`/`Type` fallback below. Only trusted when it's a real result;
    a captured `None` falls through to the logic below instead, since
    `Entry`/`Type` info didn't exist yet at capture time and might still
    resolve something now.
    """
    static_type = getattr(node, "_stubgen_static_type", None)
    if static_type is not None:
        return static_type

    try:
        base_type = node.base_type
        if isinstance(base_type, _ConstOrVolatileTypeNode):
            base_type = base_type.base_type
    except AttributeError:
        if isinstance(node, ExprNodes.ExprNode):
            return unparse_expr(node)
        base_type = None

    if base_type is None:
        return _extract_resolved_type(node)
    return _extract_structural_type(node, base_type, is_ptr)


def _declarator_name(
    decl: Nodes.CNameDeclaratorNode | Nodes.CPtrDeclaratorNode | _ConstDeclaratorNode,
) -> str | None:
    """Recursively unwrap pointer/const/func declarators to reach the name."""
    if isinstance(
        decl,
        (
            Nodes.CPtrDeclaratorNode,
            _ConstDeclaratorNode,
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


def get_cdef_variables(
    node: Nodes.CVarDefNode | Nodes.PropertyNode,
) -> list[tuple[str, str | None]]:
    """Return ``(name, type)`` pairs for every declarator in a cdef statement.

    A single ``cdef public int x, y, z`` node can contain multiple declarators.
    Fixed-size array types (``char[N]``, ``int[N][M]``) are resolved via the
    base_type's ``TemplatedTypeNode``; pointer declarators on ``char`` emit
    ``"bytes"``; function-pointer declarators emit ``"Callable"``.

    Also accepts a ``PropertyNode`` -- what ``cdef public``/``cdef readonly``
    attributes on an extension type become once real declaration analysis
    has run (``AnalyseDeclarationsTransform`` synthesizes a property with
    ``__get__``/``__set__`` in place of the original ``CVarDefNode``), and
    the exact same shape a real, source-level ``@property``-decorated
    ``def`` method takes too. Always exactly one name; the type comes from
    the property's own ``__get__`` return annotation when the source wrote
    one, or from the property's ``Entry`` otherwise (see below).
    """
    if isinstance(node, Nodes.PropertyNode):
        # A `@property`-decorated `def` method becomes this exact same
        # `PropertyNode` shape once real declaration analysis runs --
        # not just a `cdef public`/`cdef readonly` C attribute -- but
        # its `entry.type` is always the generic `PyObjectType`
        # ("object"): Cython has no reason to track anything more
        # specific for a plain Python property at the Entry/Type level.
        # The real declared type, when the source wrote one (`-> int`),
        # survives instead on the synthesized `__get__` method's own
        # `return_type_annotation` -- checked first and preferred over
        # `entry.type` whenever present.
        entry = node.entry
        type_name = render_pyrex_type(entry.type) if entry else None
        getter = next(
            (
                s
                for s in getattr(node.body, "stats", ())
                if getattr(s, "name", None) == "__get__"
            ),
            None,
        )
        annotation_node = getattr(getter, "return_type_annotation", None)
        if annotation_node is not None:
            type_name = parameterize_builtin_generic(
                decode_or_pass(annotation_node.string.value)
            )
        return [(node.name, type_name)]

    accepted = (
        Nodes.CNameDeclaratorNode,
        Nodes.CPtrDeclaratorNode,
        _ConstDeclaratorNode,
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


def _fused_member_name(node: Nodes.Node) -> str | None:
    """Return a dotted type name for a simple or templated fused-type
    member node, or None.

    Used only for ``ctypedef fused`` member type nodes (``node.types`` on
    a ``FusedTypeNode``), which are always a simple or (rarely) templated
    base-type node -- never anything `extract_type_from_base_type`'s
    fuller machinery is needed for. Kept name-only (no full type
    extraction) deliberately: this runs pre-pipeline, in
    `static_annotations.capture_static_types`, specifically so it works from raw syntax
    alone -- no `Entry`/`Type` resolution required -- see that function's
    docstring for why a fused type's own members can't always wait for
    resolution (a member naming an extension type declared only in the
    companion `.pyx`, not the `.pxd` doing the fused declaration, never
    resolves during the `.pxd`'s own, separate pipeline run).
    """
    while isinstance(node, Nodes.TemplatedTypeNode):
        node = node.base_type_node
    if isinstance(node, Nodes.CSimpleBaseTypeNode):
        return ".".join(node.module_path + [node.name])
    return None


def _cvardef_declarator_name(declarator) -> str | None:
    """The name a `CVarDefNode`'s declarator declares, unwrapping a
    pointer/const wrapper -- same shape as `signature._to_argument`'s
    equivalent unwrapping for an argument declarator, used here for
    `static_annotations.capture_static_types`'s property-type-by-name capture.
    """
    while isinstance(declarator, (Nodes.CPtrDeclaratorNode, _ConstDeclaratorNode)):
        declarator = declarator.base
    return getattr(declarator, "name", None) or None


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
        scalar = None if name is None else CYTHON_TO_NUMPY_SCALAR.get(name)
        if scalar:
            return f"numpy.typing.NDArray[numpy.{scalar}]"
    return "memoryview"
