"""``ctypedef fused`` resolution: collecting fused type definitions and
resolving fused-typed annotations (including fused-typed memoryviews) in
a signature to their final Python form (a ``TypeVar``, a ``|``-union, or
``object``).
"""

from __future__ import annotations

from Cython.Compiler import Nodes

from ..analysis.visitor import ScopeVisitor
from ..models.pyi_elements import PyiAssignment, PyiFusedType, PyiSignature
from ..postprocessing.normalize_names import _CYTHON_TRANSLATIONS
from .type_parsing import _CYTHON_TO_NUMPY_SCALAR, render_pyrex_type


def convert_fused_types(visitor: ScopeVisitor) -> dict[str, PyiFusedType]:
    """Collect ``ctypedef fused`` definitions visible in this scope.

    A ``cdef fused`` declaration vanishes from ``body.stats`` once real
    declaration analysis runs -- same as an enum or struct/union --
    so ``visitor.fused_types`` (the structural walk) is empty in
    practice by the time this runs. The structural path below
    still runs first in case a future Cython version (or an
    earlier pipeline stage) leaves the node intact.

    Two fallbacks recover the common case where it doesn't, tried in
    order:

    1. The pre-pipeline snapshot ``capture_static_types`` stashed on
       the enclosing module/class node (``_stubgen_static_fused_members``
       -- raw member name strings, never resolved ``Type``s). Needed
       specifically for a ``.pxd``-declared fused type whose members
       name an extension type declared only in the matching ``.pyx``:
       by the time the ``.pxd``'s own separate pipeline run resolves
       the fused type's members, that extension type doesn't exist
       yet in scope, so ``entry.type.types`` (fallback 2) ends up
       holding ``PyrexTypes.ErrorType`` placeholders instead. The
       raw syntax has no such problem.
    2. ``scope.entries`` directly, reading each fused type's member
       ``PyrexType``s straight off ``entry.type.types`` -- the same
       pattern already used for struct/union members via
       ``entry.type.scope``. Used whenever no snapshot exists for a
       given name (e.g. a fused type inherited via a real cross-file
       ``cimport`` rather than the ``.pxd``/``.pyx`` pair merge, where
       member resolution isn't subject to the same ordering problem).
    """
    fused_types: dict[str, PyiFusedType] = {}
    for node in visitor.fused_types:
        name = node.name
        type_nodes = node.types
        raw_names = [_type_name(type_node) for type_node in type_nodes]
        concrete_by_raw = {
            _CYTHON_TRANSLATIONS.get(raw_name, raw_name): raw_name
            for raw_name in raw_names
            if raw_name is not None
        }
        concrete_types = tuple(concrete_by_raw.keys())
        numpy_scalars = tuple(
            _CYTHON_TO_NUMPY_SCALAR.get(raw_name)
            for raw_name in concrete_by_raw.values()
        )
        fused_types[name] = PyiFusedType(name, concrete_types, numpy_scalars)

    static_fused_members = (
        getattr(visitor.node, "_stubgen_static_fused_members", None) or {}
    )
    for name, member_names in static_fused_members.items():
        if name in fused_types:
            continue
        concrete_by_raw = {
            _CYTHON_TRANSLATIONS.get(member_name, member_name): member_name
            for member_name in member_names
        }
        concrete_types = tuple(concrete_by_raw.keys())
        if concrete_types:
            numpy_scalars = tuple(
                _CYTHON_TO_NUMPY_SCALAR.get(raw_name)
                for raw_name in concrete_by_raw.values()
            )
            fused_types[name] = PyiFusedType(name, concrete_types, numpy_scalars)

    scope = getattr(visitor.node, "scope", None)
    if scope is not None:
        for name, entry in scope.entries.items():
            if name in fused_types:
                continue
            if not (entry.is_type and getattr(entry.type, "is_fused", False)):
                continue
            concrete_by_member = {}
            for member in entry.type.types:
                rendered = render_pyrex_type(member)
                if rendered is not None and rendered not in concrete_by_member:
                    concrete_by_member[rendered] = member
            concrete_types = tuple(concrete_by_member.keys())
            if concrete_types:
                numpy_scalars = tuple(
                    _CYTHON_TO_NUMPY_SCALAR.get(getattr(member, "name", None))
                    for member in concrete_by_member.values()
                )
                fused_types[name] = PyiFusedType(name, concrete_types, numpy_scalars)

    return fused_types


def convert_fused_type(fused_type: PyiFusedType) -> PyiAssignment:
    concrete_types = ", ".join(fused_type.concrete_types)
    return PyiAssignment(
        f'{fused_type.name} = typing.TypeVar("{fused_type.name}", {concrete_types})',
        name=fused_type.name,
    )


def _restore_fused_memoryview_annotations(
    signature: PyiSignature,
    node: Nodes.CFuncDefNode | Nodes.DefNode,
    fused_types: dict[str, PyiFusedType],
) -> PyiSignature:
    """Restore fused typedef names on memoryview annotations flattened to "memoryview".

    Fused types aren't threaded through the memoryview-extraction path, so a
    fused-typed memoryview loses its typedef name and comes out as the
    literal string "memoryview". This walks the AST alongside the extracted
    signature and puts the fused typedef name back wherever that happened,
    so `_resolve_fused_signature` can resolve it properly.
    """
    if not fused_types:
        return signature

    if isinstance(node, Nodes.CFuncDefNode):
        arg_nodes = getattr(getattr(node, "declarator", None), "args", [])
    else:
        arg_nodes = getattr(node, "args", [])
    for arg, arg_node in zip(signature.args, arg_nodes):
        fused_name = _fused_memoryview_name(arg_node, fused_types)
        if arg.annotation == "memoryview" and fused_name is not None:
            arg.annotation = fused_name
            # Marks this occurrence as array-shaped for
            # `_resolve_fused_signature`, since by that point the
            # annotation is just the bare typedef name ("numeric"),
            # indistinguishable from a genuinely scalar occurrence.
            arg._stubgen_fused_was_memoryview = True

    fused_name = _fused_memoryview_name(node, fused_types)
    if signature.return_type == "memoryview" and fused_name is not None:
        signature.return_type = fused_name
        signature._stubgen_fused_return_was_memoryview = True

    return signature


def _fused_memoryview_name(
    node: Nodes.Node, fused_types: dict[str, PyiFusedType]
) -> str | None:
    """Return the fused typedef name backing a memoryview node, or None if not fused."""
    base_type = getattr(node, "base_type", None)
    if not isinstance(base_type, Nodes.MemoryViewSliceTypeNode):
        return None

    scalar = getattr(base_type, "base_type_node", None)
    name = getattr(scalar, "name", None)
    return name if name in fused_types else None


def _resolve_fused_signature(
    signature: PyiSignature, fused_types: dict[str, PyiFusedType]
) -> PyiSignature:
    """Rewrite a signature's fused-type annotations to their resolved Python form.

    A fused typedef resolves to a `TypeVar` when it appears in both a
    parameter and the return, to a `|`-union when it appears in one
    parameter only, or to `object` when its member set includes `object`.
    This is a whole-signature property, so it's resolved here as a
    post-processing pass over the already-generated annotation strings
    rather than while each argument is being generated.
    """
    usage: dict[str, tuple[int, bool]] = {}
    for name in fused_types:
        param_count = sum(
            _annotation_uses_name(arg.annotation, name) for arg in signature.args
        )
        used_as_return = _annotation_uses_name(signature.return_type, name)
        if param_count or used_as_return:
            usage[name] = (param_count, used_as_return)
    for arg in signature.args:
        if arg.annotation is not None:
            arg.annotation = _resolved_fused_annotation(
                arg.annotation,
                usage,
                fused_types,
                is_memoryview=getattr(arg, "_stubgen_fused_was_memoryview", False),
            )
    if signature.return_type is not None:
        signature.return_type = _resolved_fused_annotation(
            signature.return_type,
            usage,
            fused_types,
            is_memoryview=getattr(
                signature, "_stubgen_fused_return_was_memoryview", False
            ),
        )
    return signature


def _fused_memoryview_union(value: PyiFusedType) -> str:
    """Render a memoryview-typed fused type's members as an ``NDArray`` union.

    Each member renders as ``numpy.typing.NDArray[numpy.<scalar>]`` rather
    than a bare Python scalar, since the argument/return this backs is a
    typed memoryview, not a scalar. Falls back to the unspecific but
    non-misleading ``memoryview`` for a member with no numpy scalar
    equivalent (extension type, ``object``, etc.).
    """
    members = dict.fromkeys(
        f"numpy.typing.NDArray[numpy.{scalar}]" if scalar is not None else "memoryview"
        for scalar in (value.numpy_scalars or (None,) * len(value.concrete_types))
    )
    return " | ".join(members)


def _resolved_fused_annotation(
    annotation: str,
    usage: dict[str, tuple[int, bool]],
    fused_types: dict[str, PyiFusedType],
    *,
    is_memoryview: bool = False,
) -> str:
    """Resolve a single annotation string using per-name fused usage and definitions."""
    resolved = annotation
    for name, value in fused_types.items():
        if not _annotation_uses_name(resolved, name):
            continue
        param_count, used_as_return = usage[name]
        replacement = name
        if "object" in value.concrete_types:
            replacement = "object"
        elif (not used_as_return and param_count <= 1) or param_count == 0:
            replacement = (
                _fused_memoryview_union(value)
                if is_memoryview
                else " | ".join(value.concrete_types)
            )
        elif is_memoryview:
            # Shared with a non-memoryview position, which would normally
            # resolve to a scalar-bound TypeVar -- but that TypeVar
            # doesn't type-check against NDArray, and there's no clean
            # way to express "array, same dtype as this other array" for
            # a fused type. Falls back to the unspecific but honest
            # `memoryview` instead of the misleading scalar `numeric`.
            replacement = "memoryview"
        resolved = " | ".join(
            replacement if part == name else part
            for part in _annotation_parts(resolved)
        )
    return resolved


def _annotation_uses_name(annotation: str | None, name: str) -> bool:
    """Return True if the given name appears as a part of the (union) annotation."""
    if annotation is None:
        return False
    return name in _annotation_parts(annotation)


def _annotation_parts(annotation: str) -> list[str]:
    """Split a union annotation string into its stripped alternative parts."""
    return [part.strip() for part in annotation.split("|")]


def _type_name(node: Nodes.Node) -> str | None:
    """Return a dotted type name for a simple or templated base-type node, or None."""
    while isinstance(node, Nodes.TemplatedTypeNode):
        node = node.base_type_node
    if isinstance(node, Nodes.CSimpleBaseTypeNode):
        return ".".join(node.module_path + [node.name])
    return None
