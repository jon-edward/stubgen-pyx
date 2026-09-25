"""Type parsing helpers for Cython AST nodes."""

from __future__ import annotations

import logging

from Cython.Compiler import ExprNodes, Nodes
from Cython.Compiler import PyrexTypes as _PyrexTypes
from Cython.Compiler.ModuleNode import ModuleNode as _ModuleNode

from ..logging_utils import with_debug_fallback
from .unparse import unparse_expr

_logger = logging.getLogger(__name__)

# Cython 3.3 renamed the const/volatile declarator and base-type node
# classes (`CConstDeclaratorNode` -> `CQualifierDeclaratorNode`,
# `CConstOrVolatileTypeNode` -> `CQualifierTypeNode`, the latter now
# also covering `restrict`), keeping the same attribute shape
# (`.base`/`.base_type`, `.is_const`, `.is_volatile`). Resolved once
# here so the rest of this module can use one name regardless of which
# Cython version (>=3.0) is installed.
_ConstDeclaratorNode = getattr(Nodes, "CQualifierDeclaratorNode", None) or Nodes.CConstDeclaratorNode
_ConstOrVolatileTypeNode = getattr(Nodes, "CQualifierTypeNode", None) or Nodes.CConstOrVolatileTypeNode

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


def render_pyrex_type(t: _PyrexTypes.PyrexType | None, *, _depth: int = 0) -> str | None:
    """Render a resolved ``PyrexTypes.Type`` as a Python annotation string.

    The Entry/Type-based counterpart to ``extract_type_from_base_type``,
    for the (common, post-analysis) case where there's no raw
    ``base_type`` AST node left to walk structurally -- see the design
    doc, "FileDescriptors instead of StringDescriptors": once real
    declaration analysis runs, many declarations (module/class-level
    variables with no initializer, ``cdef enum``, ``cdef struct``/
    ``union``) are removed from the tree entirely and exist only as
    ``Entry`` objects, each carrying a resolved ``.type``. Follows the
    same output conventions as the structural path (``char *`` ->
    ``"bytes"``, fixed arrays -> ``list[T]``, C++ templates ->
    ``"Base[T1, T2]"``, typed memoryviews -> ``numpy.typing.NDArray[...]``)
    so callers don't see a difference depending on which path resolved a
    given declaration.
    """
    if t is None:
        return None

    if getattr(t, "is_cv_qualified", False):
        # `const`/`volatile` (both represented by `CConstOrVolatileType`,
        # distinguished by `is_const`/`is_volatile`) have no Python-level
        # equivalent -- render the underlying type, same as the
        # structural path (`extract_type_from_base_type` unwraps
        # `CConstOrVolatileTypeNode` the same way).
        return render_pyrex_type(t.cv_base_type, _depth=_depth)

    if t.is_void:
        return "None"

    if t.is_ptr:
        base = t.base_type
        if base is _PyrexTypes.c_char_type:
            return "bytes"
        if base.is_void:
            return "typing.Any"
        if getattr(base, "is_cfunction", False):
            return _render_cfunction_type(base, _depth=_depth)
        # Cython pointers to anything else aren't meaningfully
        # Python-representable; matches the structural path, which never
        # gives pointer-ness special treatment beyond char*/void* either
        # (see `extract_type_from_base_type`).
        return render_pyrex_type(base, _depth=_depth + 1)

    if t.is_array:
        if t.base_type is _PyrexTypes.c_char_type:
            return "bytes"
        inner = render_pyrex_type(t.base_type, _depth=_depth + 1)
        return f"list[{inner}]" if inner is not None else None

    if getattr(t, "is_ctuple", False):
        parts = [
            with_debug_fallback(
                render_pyrex_type(c, _depth=_depth + 1),
                "object",
                lambda c_idx_=c_idx: f"Replaced tuple component at index {c_idx_} with 'object'",
            )
            for c_idx, c in enumerate(t.components)
        ]
        return f"tuple[{', '.join(parts)}]"

    if getattr(t, "is_memoryviewslice", False):
        dtype_name = str(t.dtype) if t.dtype is not None else None
        scalar = None if dtype_name is None else _CYTHON_TO_NUMPY_SCALAR.get(dtype_name)
        if scalar:
            return f"numpy.typing.NDArray[numpy.{scalar}]"
        return "memoryview"

    if getattr(t, "is_cpp_class", False) and getattr(t, "templates", None):
        base = t.name
        parts = [
            with_debug_fallback(
                render_pyrex_type(a, _depth=_depth + 1),
                "_typeshed.Incomplete",
                lambda a_idx_=a_idx: (
                    f"Replaced template argument of {base} at index {a_idx_} with '_typeshed.Incomplete'"
                ),
            )
            for a_idx, a in enumerate(t.templates)
        ]
        return f"{base}[{', '.join(parts)}]"

    # Struct/union, enum (including a C++11 scoped `enum class`, whose
    # `is_enum` is `False` -- `is_cpp_enum` instead), extension type
    # ("cdef class"), plain (non-template) cpp class: all expose a plain
    # `.name` matching what the structural path would have produced from
    # the declaration's own base-type name.
    name = getattr(t, "name", None)
    if name is not None and (
        getattr(t, "is_struct_or_union", False)
        or getattr(t, "is_enum", False)
        or getattr(t, "is_cpp_enum", False)
        or getattr(t, "is_extension_type", False)
        or getattr(t, "is_cpp_class", False)
        or getattr(t, "is_fused", False)
    ):
        return name

    if getattr(t, "is_cfunction", False):
        return _render_cfunction_type(t, _depth=_depth)

    if t.is_pyobject or t.is_numeric or t.is_string:
        # `py_type_name()` already returns Cython's own best Python-facing
        # name for these (int/float/bool/str/bytes/object/...); no need
        # to hand-roll a second mapping that would inevitably drift from
        # Cython's own as new C types are added.
        py_name = t.py_type_name()
        return parameterize_builtin_generic(py_name)

    return None


def _render_cfunction_type(t: _PyrexTypes.CFuncType, *, _depth: int) -> str:
    """Render a resolved ``CFuncType`` (a function pointer's pointee, typically) as ``Callable[[...], ...]``."""
    args = [
        with_debug_fallback(
            render_pyrex_type(arg.type, _depth=_depth + 1),
            "_typeshed.Incomplete",
            lambda arg_idx_=arg_idx: f"Replaced argument {arg_idx_} type with '_typeshed.Incomplete'",
        )
        for arg_idx, arg in enumerate(t.args)
    ]
    return_type = with_debug_fallback(
        render_pyrex_type(t.return_type, _depth=_depth + 1),
        "_typeshed.Incomplete",
        lambda: "Replaced return type with '_typeshed.Incomplete'",
    )
    return f"typing.Callable[[{', '.join(args)}], {return_type}]"


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
            (s for s in getattr(node.body, "stats", ()) if getattr(s, "name", None) == "__get__"),
            None,
        )
        annotation_node = getattr(getter, "return_type_annotation", None)
        if annotation_node is not None:
            from .signature import _decode_or_pass  # local: avoids a circular import with signature.py

            type_name = parameterize_builtin_generic(
                _decode_or_pass(annotation_node.string.value)
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
    `capture_static_types`, specifically so it works from raw syntax
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
    `capture_static_types`'s property-type-by-name capture.
    """
    while isinstance(declarator, (Nodes.CPtrDeclaratorNode, _ConstDeclaratorNode)):
        declarator = declarator.base
    return getattr(declarator, "name", None) or None


def capture_static_types(tree) -> None:
    """Snapshot pre-pipeline structural info the pipeline would otherwise destroy.

    Some declarations keep their AST node through the whole pipeline
    (`AnalyseDeclarationsTransform` mutates them in place rather than
    replacing them -- same object, before and after, for `CFuncDefNode`/
    `CArgDeclNode`), but have attributes the (still node-based) converter
    needs cleared as part of normal analysis:

    - ``.base_type``: cleared once a real ``PyrexTypes.Type`` is
      resolved for the declaration. For most cases the resolved `Type`
      is a perfectly good replacement (see `render_pyrex_type`); it
      isn't always -- Cython doesn't retain a Python-generic's type
      parameters at the `Type` level at all (`list[int]` used directly
      as a Cython type resolves to a `BuiltinTypeConstructorObjectType`
      with no trace of the `int`). Entries genuinely don't get you 100%
      of the way there for these.
    - ``.decorators``: a real Python-level decorator (`@functools.
      singledispatch`, say) is rewritten by `AnalyseDeclarationsTransform`
      into a bare, decorator-less `DefNode` plus a separate
      `f = the_decorator(f)` assignment. That's the right shape for
      code generation; for stub generation it means losing every
      decorator in the output, since `get_decorators` renders
      `node.decorators` directly as `@...` lines.
    - Bare annotated attributes with no value (`x: int`, no `= ...`) in a
      module or plain (non-``cdef``) class body: each one parses as its
      own `ExprStatNode(NameNode(annotation=...))`, but
      `AnalyseDeclarationsTransform` folds *all* of them together into a
      single synthetic `__annotations__ = {...}` dict assignment.
      Individually-typed attributes are exactly what a stub needs; a
      single dict-valued assignment named `__annotations__` is both
      useless as a `.pyi` member and, for a `TypedDict` subclass
      specifically, invalid syntax there.

    - A ``ctypedef fused`` declaration's member types (``node.types`` on
      the ``FusedTypeNode``): captured as plain name strings, not
      resolved ``Type``s. A fused type declared in a companion ``.pxd``
      whose members name an extension type declared only in the
      matching ``.pyx`` (the common case for a forward-declared
      ``cpdef``) never resolves during the ``.pxd``'s own, separate
      pipeline run: those member lookups fail and the ``Entry``'s
      ``.type.types`` ends up holding ``PyrexTypes.ErrorType``
      placeholders instead. The raw syntax never has this problem -- a
      member's name is just its name -- so it's captured here instead
      of relying on resolution to ever succeed.

    The original AST has all of these, right up until the pipeline
    clears/folds them. Called once, right after parsing and before
    running the pipeline (see `parsing/parser.py`), on the still-intact
    raw tree. Walks every node reachable via ``child_attrs`` and stashes
    what it finds as private ``_stubgen_static_type``/
    ``_stubgen_static_decorators``/``_stubgen_static_annotations``/
    ``_stubgen_static_fused_members`` attributes directly on the node.
    `extract_type_from_base_type`, `source_extraction.get_decorators`,
    and `Converter._convert_declared_entries`/`convert_fused_types`
    check for these first.

    Entry/Type resolution stays the *only* option for declarations that
    are removed from the tree entirely (uninitialized variables, enums,
    structs/unions) -- there's no node left to capture anything from,
    but those are always plain C types with no decorators to lose, so
    `render_pyrex_type` alone is sufficient for them.
    """
    seen: set[int] = set()
    _capture_static_types_recursive(tree, None, seen)


def _capture_static_types_recursive(node, enclosing, seen: set[int]) -> None:
    if node is None or id(node) in seen:
        return
    seen.add(id(node))

    if hasattr(node, "base_type"):
        try:
            node._stubgen_static_type = extract_type_from_base_type(node)
        except AttributeError:
            pass

    if isinstance(node, Nodes.CVarDefNode) and enclosing is not None:
        # A `cdef public`/`cdef readonly` attribute on a `cdef class` is
        # replaced entirely by a synthesized `PropertyNode`
        # (`__get__`/`__set__` in place of the original declaration --
        # see `type_parsing.get_cdef_variables`'s docstring) once real
        # declaration analysis runs: `node._stubgen_static_type`,
        # captured on *this* CVarDefNode just above, is simply gone by
        # the time conversion runs, since that's a different, later
        # -created object with no memory of it. Matters specifically
        # for a qualified type (`cdef public mod.Foo item`, `mod`
        # cimported): resolving the resulting PropertyNode's `entry.
        # type` only ever gives the bare class name (`Foo`) -- a
        # `PyExtensionType`'s own `.name` carries no memory of which
        # module it was accessed through, that's a property of the
        # source syntax, not the resolved type -- and otherwise
        # renders as `item: Foo` (then trimmed to `_typeshed.Incomplete`
        # by `postprocessing/trim_not_defined.py`, since bare `Foo` is
        # never actually a defined name in the stub). Stashed here, on
        # the nearest enclosing class/module node (which does survive),
        # keyed by declared name, for `Converter.convert_scope` to
        # prefer over the entries-based fallback when present.
        type_str = getattr(node, "_stubgen_static_type", None)
        if type_str is not None:
            for declarator in getattr(node, "declarators", None) or ():
                decl_name = _cvardef_declarator_name(declarator)
                if decl_name is None:
                    continue
                property_types = getattr(enclosing, "_stubgen_static_property_types", None)
                if property_types is None:
                    property_types = enclosing._stubgen_static_property_types = {}
                property_types[decl_name] = type_str

    if isinstance(node, Nodes.CArgDeclNode):
        declarator = getattr(node, "declarator", None)
        # `_declarator_name` unwraps every declarator layer (pointer,
        # const, func, array) down to the name. A single-level unwrap
        # (checking only `CPtrDeclaratorNode`, reading `.base.name`)
        # returns None for a const-qualified pointer (`int* const p`),
        # since `declarator.base` is then a `CConstDeclaratorNode` with
        # no `.name` of its own -- which would falsely mark a real,
        # named argument as "bare" below.
        declared_name = _declarator_name(declarator)
        if not declared_name:
            # A bare, unannotated argument (`self`, `cls`, or a plain
            # untyped positional like `x` in `def f(x=None)`/
            # `cpdef f(x=None)`) is grammatically indistinguishable, at
            # parse time, from a type-only declaration with no variable
            # name: Cython parses it with an *empty* `declarator.name`
            # (or `declarator.base.name`, for a pointer arg) and the
            # identifier itself sitting on `base_type.name` instead, for
            # both `self` specifically (self-arg analysis) and any
            # other bare positional (general type inference). Either
            # kind of later analysis backfills the declarator's name
            # from that same identifier, which would otherwise make
            # `signature._to_argument`'s own handling of this same
            # ambiguity (checking whether `name` is falsy) stop
            # triggering post-pipeline, and the leftover `base_type` --
            # still holding that same bogus identifier -- get extracted
            # as a real type annotation instead (rendered downstream as
            # e.g. `self: _typeshed.Incomplete` or `x: x | None`).
            # Captured here, pre-pipeline, before that backfill
            # happens, so it's never ambiguous later.
            node._stubgen_bare_identifier_arg = True

    decorators = getattr(node, "decorators", None)
    if decorators:
        node._stubgen_static_decorators = list(decorators)

    if (
        isinstance(node, Nodes.ExprStatNode)
        and isinstance(node.expr, ExprNodes.NameNode)
        and node.expr.annotation is not None
        and enclosing is not None
    ):
        annotations = getattr(enclosing, "_stubgen_static_annotations", None)
        if annotations is None:
            annotations = enclosing._stubgen_static_annotations = []
        annotations.append((node.expr.name, unparse_expr(node.expr.annotation.expr)))

    if isinstance(node, Nodes.FusedTypeNode) and enclosing is not None:
        member_names = tuple(
            name
            for name in (_fused_member_name(t) for t in node.types)
            if name is not None
        )
        if member_names:
            fused_members = getattr(enclosing, "_stubgen_static_fused_members", None)
            if fused_members is None:
                fused_members = enclosing._stubgen_static_fused_members = {}
            fused_members[node.name] = member_names

    # Track the nearest module/class body as the target for annotation
    # captures -- `Nodes.ModuleNode`/`Nodes.PyClassDefNode`/
    # `Nodes.CClassDefNode` are the scopes `_convert_declared_entries`
    # can actually attach synthesized assignments to.
    next_enclosing = (
        node
        if isinstance(node, (_ModuleNode, Nodes.PyClassDefNode, Nodes.CClassDefNode))
        else enclosing
    )

    for attr_name in getattr(node, "child_attrs", None) or ():
        child = getattr(node, attr_name, None)
        if isinstance(child, list):
            for item in child:
                _capture_static_types_recursive(item, next_enclosing, seen)
        else:
            _capture_static_types_recursive(child, next_enclosing, seen)


def extract_type_from_base_type(node, is_ptr: bool = False) -> str | None:
    """Extract a type annotation string from a base_type node.

    Handles plain named types, pointer types (``char *`` -> ``bytes``),
    tuple types, C++ templated types, fixed-size C arrays, and typed
    memoryviews.

    Checks for a value stashed by `capture_static_types` first: some
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
    # Trust a captured value only when it's a real, positive result --
    # `capture_static_types` runs before `Entry`/`Type` info exists, so a
    # `None` there just means structural extraction alone found nothing
    # *at that point*; the `Entry`/`Type` fallback below might still
    # succeed once the pipeline has run, and should get the chance to.
    static_type = getattr(node, "_stubgen_static_type", None)
    if static_type is not None:
        return static_type

    try:
        base_type = node.base_type
        if isinstance(base_type, _ConstOrVolatileTypeNode):
            base_type = base_type.base_type
    except AttributeError:
        base_type = None
        if isinstance(node, ExprNodes.ExprNode):
            return unparse_expr(node)

    if base_type is None:
        resolved_type = getattr(node, "type", None) or getattr(
            getattr(node, "entry", None), "type", None
        )
        # A CFuncDefNode/DefNode's own `.type` is its *whole* function
        # signature, not a value type -- never the right thing to render
        # here. Leave those to the caller (e.g.
        # `signature._get_return_type_annotation`, which narrows to
        # `.type.return_type` itself) rather than rendering the callable.
        #
        # `py_object_type` (Cython's exact singleton for "no C type was
        # ever written here") is likewise not real fallback info: a
        # genuinely untyped `cdef`/`cpdef` argument (`def f(x): ...`,
        # no annotation at all) resolves to this same singleton, just
        # like an explicitly-typed one would -- there's nothing in the
        # resolved `Type` that distinguishes "explicitly typed as
        # object" from "not typed at all". Rendering "object" for a
        # plain, unannotated argument would be new, unwarranted
        # information the source never actually stated -- matches the
        # pre-migration structural extraction, which simply had no node
        # to find for an untyped argument and produced no annotation.
        # Also the routine outcome for `self`/`cls` in a plain Python
        # class (as opposed to a `cdef class`): `is_self_arg` is never
        # set for it (Cython has no need to give a non-extension-type
        # `self` a real C-level type), so it falls
        # through to here rather than the dedicated, silent handling
        # `signature._to_argument` has for a `cdef class`'s `self`/`cls`
        # -- and resolves to this exact singleton, same reasoning as
        # any other untyped argument.
        skip_reason_is_expected = resolved_type is None or (
            resolved_type is _PyrexTypes.py_object_type
            or isinstance(node, (Nodes.CFuncDefNode, Nodes.DefNode))
        )
        if not skip_reason_is_expected:
            # Post-analysis, several declaration kinds no longer carry a
            # `base_type` AST node at all -- either because `node` itself
            # has no `.base_type` attribute (caught above), or because it
            # does but analysis has since cleared it, as happens for a
            # `CArgDeclNode` whose real type lives on `.type` instead once
            # resolved (see `render_pyrex_type`'s docstring). Fall back to
            # whatever real `Entry`/`Type` info survived instead of
            # giving up.
            rendered = render_pyrex_type(resolved_type)
            if rendered is not None:
                return rendered
            # `resolved_type` existed but `render_pyrex_type` couldn't
            # render it -- a genuine "don't know what this is" case,
            # worth logging (unlike the expected cases above, which
            # return None just below without logging anything).
            _logger.debug("Unknown base type: %s", type(node).__name__)
            return None
        if resolved_type is None:
            # Nothing was found at all -- also genuinely unexpected,
            # and worth logging, unlike the `py_object_type`/function
            # -node cases just above (both routine, not logged).
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
