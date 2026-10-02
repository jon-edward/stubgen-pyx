"""Static annotation recovery: capture pre-pipeline, convert post-pipeline.

`AnalyseDeclarationsTransform` destroys the individual, ordered type
declarations this module needs (folding bare-annotated attributes into a
single synthetic ``__annotations__`` dict, stripping decorators, clearing
resolved-away structural type info). `capture_static_types` runs once,
immediately after parsing and before that pass, and stashes what it finds
as private ``_stubgen_*`` attributes directly on the still-intact AST
nodes; `convert_declared_entries` and friends run after the full pipeline
and turn both those stashed attributes and whatever plain `Symtab.Entry`
objects survive into `PyiAssignment`/`PyiEnum`/`PyiClass`/`PyiFunction`
objects for a scope.
"""

from __future__ import annotations

from Cython.Compiler import ExprNodes, Nodes
from Cython.Compiler.ModuleNode import ModuleNode as _ModuleNode

from ..analysis.visitor import ScopeVisitor
from ..logging_utils import with_debug_fallback
from ..models.pyi_elements import (
    PyiArgument,
    PyiAssignment,
    PyiClass,
    PyiEnum,
    PyiFunction,
    PyiSignature,
)
from .ctypedef_aliases import _substitute_ctypedef_aliases
from .declarations import convert_struct_or_union_type
from .pyrex_types import render_pyrex_type
from .type_parsing import (
    _cvardef_declarator_name,
    _declarator_name,
    _fused_member_name,
    extract_type_from_base_type,
)
from .unparse import unparse_expr


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
    and `convert_declared_entries`/`convert_fused_types` check for
    these first.

    Entry/Type resolution stays the *only* option for declarations that
    are removed from the tree entirely (uninitialized variables, enums,
    structs/unions) -- there's no node left to capture anything from,
    but those are always plain C types with no decorators to lose, so
    `render_pyrex_type` alone is sufficient for them.
    """
    seen: set[int] = set()
    _capture_static_types_recursive(tree, None, seen)


def _capture_static_type(node) -> None:
    if hasattr(node, "base_type"):
        try:
            node._stubgen_static_type = extract_type_from_base_type(node)
        except AttributeError:
            pass


def _capture_property_type(node, enclosing) -> None:
    if isinstance(node, Nodes.CVarDefNode) and enclosing is not None:
        type_str = getattr(node, "_stubgen_static_type", None)
        if type_str is None:
            return
        for declarator in getattr(node, "declarators", None) or ():
            decl_name = _cvardef_declarator_name(declarator)
            if decl_name is None:
                continue
            property_types = getattr(enclosing, "_stubgen_static_property_types", None)
            if property_types is None:
                property_types = enclosing._stubgen_static_property_types = {}
            property_types[decl_name] = type_str


def _set_static_python_annotation(enclosing, name: str, type_str: str) -> None:
    annotations = getattr(enclosing, "_stubgen_static_python_annotations", None)
    if annotations is None:
        annotations = enclosing._stubgen_static_python_annotations = {}
    annotations[name] = type_str


def _capture_annotated_assignment_property_type(node, enclosing) -> None:
    """Record a Python-style-annotated class attribute's own source
    annotation (``x: cython.double = 0.0``, ``pt: Point``, ``x: float |
    None = None``) into ``_stubgen_static_python_annotations``, keyed by
    name.

    This shape parses as a `SingleAssignmentNode` with an annotated
    `NameNode` target, not a `CVarDefNode` -- `_capture_property_type`
    never sees it. Once real declaration analysis runs, this attribute
    becomes a synthesized `PropertyNode` (see `ScopeVisitor.
    visit_PropertyNode`) whose `Entry.type` is only ever the single,
    concrete C/Python type Cython actually gives the attribute's storage
    slot -- never the full union/generic/plain-Python-class annotation
    the source wrote, for three different reasons `_convert_cdef_
    assignments` all resolves the same way, from this same snapshot:

    - A plain Python class (not itself a ``cdef class``) can only ever
      back a generic ``PyObject*`` slot, indistinguishable at the
      `Entry`/`Type` level from a genuinely untyped ``object`` attribute
      (``pt: Point`` -> `Entry.type` renders as plain ``object``).
    - A union that includes ``None`` alongside a real scalar type still
      gets that scalar's own specific C slot (``x: float | None`` ->
      `Entry.type` renders as ``float``, silently dropping the ``| None``
      -- and, combined with a default of ``None``, produces a `def
      __init__(..., x: float=None)` that no type checker accepts).
    - A generic parameterized over anything beyond what `PyrexTypes.Type`
      itself tracks loses those parameters entirely at the `Entry`/`Type`
      level (see `extract_type_from_base_type`'s docstring on
      ``list[int]``).

    A Cython pure-Python-mode type name (``cython.double``) also needs
    its ``cython.`` prefix stripped before use: there's no Python-level
    ``cython`` module for a `.pyi` to import, so left alone,
    `postprocessing.trim_not_defined` would find no binding for it and
    replace the whole annotation with `_typeshed.Incomplete`.
    """
    if not (
        isinstance(node, Nodes.SingleAssignmentNode)
        and isinstance(node.lhs, ExprNodes.NameNode)
        and node.lhs.annotation is not None
        and enclosing is not None
    ):
        return
    type_str = unparse_expr(node.lhs.annotation.expr)
    if type_str is None:
        return
    type_str = type_str.removeprefix("cython.")
    _set_static_python_annotation(enclosing, node.lhs.name, type_str)


def _capture_bare_identifier_arg(node) -> None:
    if isinstance(node, Nodes.CArgDeclNode):
        declarator = getattr(node, "declarator", None)
        declared_name = _declarator_name(declarator)
        if not declared_name:
            node._stubgen_bare_identifier_arg = True


def _capture_decorators(node) -> None:
    decorators = getattr(node, "decorators", None)
    if decorators:
        node._stubgen_static_decorators = list(decorators)


def _capture_annotation(node, enclosing) -> None:
    if (
        isinstance(node, Nodes.ExprStatNode)
        and isinstance(node.expr, ExprNodes.NameNode)
        and node.expr.annotation is not None
        and enclosing is not None
    ):
        type_str = unparse_expr(node.expr.annotation.expr)
        annotations = getattr(enclosing, "_stubgen_static_annotations", None)
        if annotations is None:
            annotations = enclosing._stubgen_static_annotations = []
        annotations.append((node.expr.name, type_str, node.pos[1]))
        # Also keyed by name alone (no line, overwrite-safe) for
        # `_convert_cdef_assignments` -- see
        # `_capture_annotated_assignment_property_type`'s docstring for
        # why a cdef class's own bare-annotated attribute needs this same
        # source-text override, not just a module/plain-class one.
        if type_str is not None:
            _set_static_python_annotation(enclosing, node.expr.name, type_str)


def _capture_fused_members(node, enclosing) -> None:
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


def _capture_children(node, enclosing, seen: set[int]) -> None:
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


def _capture_static_types_recursive(node, enclosing, seen: set[int]) -> None:
    if node is None or id(node) in seen:
        return
    seen.add(id(node))

    _capture_static_type(node)
    _capture_property_type(node, enclosing)
    _capture_annotated_assignment_property_type(node, enclosing)
    _capture_bare_identifier_arg(node)
    _capture_decorators(node)
    _capture_annotation(node, enclosing)
    _capture_fused_members(node, enclosing)

    _capture_children(node, enclosing, seen)


def _is_readonly_cdef_attribute(
    cdef_variable: Nodes.CVarDefNode | Nodes.PropertyNode,
) -> bool:
    """Whether a `cdef_variables` entry came from `cdef readonly` (as
    opposed to `cdef public`, or a real read/write `@property`).

    A raw `CVarDefNode` (the pre-pipeline fallback shape -- see
    `ScopeVisitor.visit_CVarDefNode`) carries its own `visibility`
    directly. A `PropertyNode` reaching `cdef_variables` at all is
    always either the synthesized pair for `cdef public` (has a
    `__set__`) or, identically shaped, a real read/write property with
    a setter -- `visit_PropertyNode` already routes every setter-less
    `PropertyNode` that isn't one of those to `property_getters`
    instead, so a `PropertyNode` with no `__set__` reaching here can
    only be the synthesized getter-only pair for `cdef readonly`.
    """
    if isinstance(cdef_variable, Nodes.PropertyNode):
        stats = getattr(cdef_variable.body, "stats", None) or ()
        return not any(getattr(s, "name", None) == "__set__" for s in stats)
    return getattr(cdef_variable, "visibility", None) == "readonly"


def _convert_declared_function(name: str, t) -> PyiFunction:
    """Convert a function-shaped Entry's `CFuncType` -- no surviving
    `CFuncDefNode`/`DefNode` (e.g. a `cpdef` declared directly inside
    `cdef extern from ...:`, with no body of its own) -- to a
    `PyiFunction`.

    No default values: `CFuncTypeArg` carries no default-value
    information at all (defaults live on the original AST node, which
    doesn't survive), so an argument with a real default renders without
    one here. A narrower result than the structural path gives, but
    still a real, usable signature rather than nothing.
    """
    args = [
        PyiArgument(
            name=arg.name or f"arg{i}",
            annotation=with_debug_fallback(
                render_pyrex_type(arg.type),
                "_typeshed.Incomplete",
                lambda name_=arg.name: f"Unable to determine type for {name_}",
            ),
        )
        for i, arg in enumerate(t.args)
    ]
    return_type = with_debug_fallback(
        render_pyrex_type(t.return_type),
        "_typeshed.Incomplete",
        lambda: f"Unable to determine return type for {name}",
    )
    return PyiFunction(
        name=name,
        is_async=False,
        signature=PyiSignature(args=args, return_type=return_type),
    )


def recover_static_annotations(
    visitor: ScopeVisitor,
    handled_names: set[str],
    ctypedef_aliases: dict[str, str] | None,
) -> list[tuple[int, PyiAssignment]]:
    """Recover bare (no-value) annotated attributes, each paired with
    its original source line.

    `AnalyseDeclarationsTransform` folds every bare-annotated
    attribute in a scope (`x: int`, no `= ...`) into a single
    synthetic `__annotations__ = {...}` dict, destroying both the
    individual declarations *and* their interleaving with whatever
    ordinary, value-having assignments (`y: int = 0`) sit alongside
    them in the same scope. The line number recovered here (from the
    pre-pipeline snapshot -- see `capture_static_types`) is what lets
    `Converter.convert_scope` re-interleave the two by original source
    position instead of dumping every recovered attribute after every
    real one, which silently reorders a dataclass's fields relative to
    its own source and can turn a valid field order into one
    `dataclasses`/mypy rejects (a default-less field put after one that
    has a default).
    """
    assignments = []
    for name, type_str, line in (
        getattr(visitor.node, "_stubgen_static_annotations", ()) or ()
    ):
        if name in handled_names:
            continue
        type_str = _substitute_ctypedef_aliases(type_str, ctypedef_aliases or {})
        assignments.append((line, PyiAssignment(f"{name}: {type_str}", name=name)))
        handled_names.add(name)
    return assignments


def convert_declared_entry(
    visitor: ScopeVisitor,
    name: str,
    entry,
    ctypedef_aliases: dict[str, str] | None,
) -> tuple[
    PyiAssignment | None,
    PyiEnum | None,
    PyiClass | None,
    PyiFunction | None,
]:
    t = entry.type
    if entry.is_type:
        if getattr(t, "is_enum", False) or getattr(t, "is_cpp_enum", False):
            if entry.create_wrapper:
                return (
                    None,
                    PyiEnum(enum_name=name, names=list(t.values)),
                    None,
                    None,
                )
            return (
                PyiAssignment(f"{name}: typing_extensions.TypeAlias = int", name=name),
                None,
                None,
                None,
            )
        if getattr(t, "is_struct_or_union", False):
            return None, None, convert_struct_or_union_type(t), None
        return None, None, None, None

    if entry.is_cfunction:
        function = _convert_declared_function(name, t) if entry.create_wrapper else None
        return None, None, None, function

    if not entry.is_variable or not (
        visitor.in_class and entry.visibility in ("public", "readonly")
    ):
        return None, None, None, None
    type_name = render_pyrex_type(t)
    resolved = with_debug_fallback(
        type_name,
        "_typeshed.Incomplete",
        lambda name_=name: f"Unable to determine type for {name_}",
    )
    resolved = _substitute_ctypedef_aliases(resolved, ctypedef_aliases or {})
    return PyiAssignment(f"{name}: {resolved}", name=name), None, None, None


def convert_declared_entries(
    visitor: ScopeVisitor,
    handled_names: set[str],
    ctypedef_aliases: dict[str, str] | None = None,
) -> tuple[
    list[PyiAssignment],
    list[PyiEnum | PyiAssignment],
    list[PyiClass],
    list[PyiFunction],
]:
    """Catch declarations with no surviving AST node in this scope.

    Once real declaration analysis runs, a declaration with no
    runtime/executable component (an uninitialized variable, a
    ``cdef enum``, a ``cdef struct``/``union``, a ``cdef fused`` type
    with no methods) is removed from the tree entirely and exists
    only as a ``Symtab.Entry``. `visitor`'s structural walk
    (``visit_CVarDefNode``, etc.) can only find what's still a node;
    this fills in the rest by walking `visitor.node.scope.entries`
    directly and skipping anything already captured structurally
    (`handled_names`) or not actually declared in this scope
    (`entry.scope is not scope` -- an imported/foreign name, already
    handled separately by `ImportVisitor`).
    """
    scope = getattr(visitor.node, "scope", None)
    if scope is None:
        return [], [], [], []

    extra_assignments: list[PyiAssignment] = []
    extra_enums: list[PyiEnum | PyiAssignment] = []
    extra_structs: list[PyiClass] = []
    extra_functions: list[PyiFunction] = []

    for name, entry in scope.entries.items():
        if name in handled_names or (name.startswith("__") and name.endswith("__")):
            continue
        if name.startswith("__pyx_"):
            continue
        if not name.isidentifier():
            # Not a name Python code could ever reference, so not
            # worth a declaration -- and, unlike `__pyx_`-prefixed
            # internal names, not reliably prefix-matchable. Seen in
            # practice: Cython auto-generates a by-value return
            # struct for an unspecialized fused `ctypedef fused`
            # ctuple (e.g. `(floating, floating)` where `floating`
            # is never resolved to a concrete type), named from
            # `PyrexTypes.c_tuple_type`'s own internal placeholder
            # cname (`"<dummy fused ctuple ...>"`, by its own
            # comment "should never end up in code") plus a
            # `_struct` suffix -- emitting it verbatim as a class
            # name produced invalid Python syntax.
            continue
        if entry.scope is not scope:
            continue

        assignment, enum, struct, function = convert_declared_entry(
            visitor, name, entry, ctypedef_aliases
        )
        if assignment is not None:
            extra_assignments.append(assignment)
        if enum is not None:
            extra_enums.append(enum)
        if struct is not None:
            extra_structs.append(struct)
        if function is not None:
            extra_functions.append(function)

    return extra_assignments, extra_enums, extra_structs, extra_functions
