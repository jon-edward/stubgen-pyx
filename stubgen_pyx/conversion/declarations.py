"""Simple, self-contained node/Entry -> PyiElement converters: structs
and unions, enums, an unsupported-C++-class fallback, plain assignments,
and a single import statement -- none of which need any state from
`Converter` itself.
"""

from __future__ import annotations

import ast
import logging
import re

from Cython.Compiler import Nodes

from ..logging_utils import with_debug_fallback
from ..models.pyi_elements import PyiAssignment, PyiClass, PyiEnum, PyiImport, PyiScope
from .pyrex_types import render_pyrex_type
from .source_extraction import get_source
from .type_parsing import (
    extract_name_and_type,
    get_cdef_variables,
    get_enum_names,
)
from .unparse import unparse_expr

_logger = logging.getLogger(__name__)

#: Rewrites a `cimport` statement's raw source text into a plain `import`
#: for the stub -- the stub is Python-facing, so `cimport foo` (Cython
#: -only syntax) becomes `import foo`, matching how the name would
#: actually be reached from a `.pyi` consumer's own import.
_CIMPORT_RE = re.compile(r"\bcimport\b")


class InvalidAssignment(Exception):
    """An invalid assignment was encountered."""


def _assignment_source(
    assignment: Nodes.AssignmentNode | Nodes.ExprStatNode, source_code: str
) -> str | None:
    out_assignment_str: str | None = None

    if isinstance(assignment, Nodes.SingleAssignmentNode):
        name = assignment.lhs.name
        expr = unparse_expr(assignment.rhs)
        if expr is not None:
            annotation = (
                assignment.lhs.annotation.string.value
                if assignment.lhs.annotation is not None
                else None
            )
            target = f"{name}: {annotation}" if annotation else name
            out_assignment_str = f"{target} = {expr}"

    if isinstance(assignment, Nodes.CTypeDefNode):
        name, typ = extract_name_and_type(assignment)
        if typ and name:
            out_assignment_str = f"{name}: typing_extensions.TypeAlias = {typ}"
        elif name:
            _logger.debug(
                "Could not extract ctypedef type: %r",
                get_source(source_code, assignment),
            )
            out_assignment_str = f"{name} = ..."
        else:
            _logger.debug(
                "Could not extract ctypedef name or type: %r",
                get_source(source_code, assignment),
            )
            return None

    return out_assignment_str or get_source(source_code, assignment)


def _parse_assignment(source: str) -> ast.Assign | ast.AnnAssign | None:
    try:
        node = ast.parse(source)
        if len(node.body) != 1 or not isinstance(
            node.body[0], (ast.Assign, ast.AnnAssign)
        ):
            raise InvalidAssignment
    except (SyntaxError, InvalidAssignment):
        _logger.debug("Could not parse assignment source: %r", source)
        return None
    return node.body[0]


def convert_import(
    node: Nodes.Node, source_code: str, raw: str | None = None
) -> PyiImport:
    """Convert a single import node to PyiImport, rewriting cimport -> import."""
    raw = raw if raw is not None else get_source(source_code, node)
    return PyiImport(_CIMPORT_RE.sub("import", raw))


def convert_struct_or_union(node: Nodes.CStructOrUnionDefNode) -> PyiClass:
    """Convert a C struct or union definition node to PyiClass."""
    node_name = node.name

    def debug_attribute(attr_name: str) -> str:
        return f"Unable to determine type for {attr_name} in {node_name}"

    attributes = []
    for attribute in node.attributes or []:
        if isinstance(attribute, Nodes.CVarDefNode):
            attributes.extend(
                [
                    PyiAssignment(
                        f"{name}: {with_debug_fallback(type_name, '_typeshed.Incomplete', lambda name_=name: debug_attribute(name_))}",
                        name=name,
                    )
                    for name, type_name in get_cdef_variables(attribute)
                ]
            )
        else:
            _logger.debug(f"Unexpected attribute type {type(attribute)} in {node_name}")
    is_union = node.kind == "union"
    keywords = {"total": "False"} if is_union else {}
    return PyiClass(
        name=node.name,
        bases=["typing.TypedDict"],
        keywords=keywords,
        scope=PyiScope(assignments=attributes),
    )


def convert_struct_or_union_type(t) -> PyiClass:
    """Convert a resolved ``CStructOrUnionType`` to a ``PyiClass``.

    The ``Entry``/``Type``-based counterpart to
    ``convert_struct_or_union`` (which walks a raw
    ``CStructOrUnionDefNode``): once real declaration analysis has
    run, a struct/union with no Python-visible companion is removed
    from the tree entirely and exists only as an ``Entry`` whose
    ``.type.scope`` holds its members. Same output convention as
    the node-based path: a ``typing.TypedDict`` subclass,
    ``total=False`` for a union (any one member may be
    absent/aliased with the others).
    """
    member_scope = t.scope
    attributes = []
    for mname, mentry in member_scope.entries.items() if member_scope else ():
        if mname.startswith("__"):
            continue
        type_name = render_pyrex_type(mentry.type)
        resolved = with_debug_fallback(
            type_name,
            "_typeshed.Incomplete",
            lambda mname_=mname: f"Unable to determine type for {mname_} in {t.name}",
        )
        attributes.append(PyiAssignment(f"{mname}: {resolved}", name=mname))
    keywords = {"total": "False"} if t.kind == "union" else {}
    return PyiClass(
        name=t.name,
        bases=["typing.TypedDict"],
        keywords=keywords,
        scope=PyiScope(assignments=attributes),
    )


def convert_cpp_class(cpp_class: Nodes.CppClassNode) -> PyiAssignment | None:
    """Convert a C++ class definition node to PyiAssignment. Currently unsupported, returns the name as an Incomplete type alias."""

    name = getattr(cpp_class, "name", None)
    _logger.debug(
        "Unsupported C++ class %s, falling back to _typeshed.Incomplete", name
    )

    return (
        PyiAssignment(
            f"{name}: typing_extensions.TypeAlias = _typeshed.Incomplete", name=name
        )
        if name
        else None
    )


def convert_assignment(
    assignment: Nodes.AssignmentNode | Nodes.ExprStatNode,
    source_code: str,
    *,
    in_class: bool = False,
) -> PyiAssignment | None:
    """Convert an assignment node to PyiAssignment, extracting type annotations."""
    source = _assignment_source(assignment, source_code)
    if source is None:
        return None
    node = _parse_assignment(source)
    if node is None:
        return None

    if in_class and _is_unhashable_hash_assignment(node):
        source = "__hash__ = None  # type: ignore[assignment]"

    name = _assignment_target_name(node)
    if name is not None and name.startswith("__pyx_"):
        return None

    return PyiAssignment(source, name=name)


def enum_alias_assignment(name: str) -> PyiAssignment:
    """The ``Name: TypeAlias = int`` stand-in for a ``cdef enum`` (no
    ``cpdef``), which has no Python-level binding of its own."""
    return PyiAssignment(f"{name}: typing_extensions.TypeAlias = int", name=name)


def is_enum_alias_assignment(assignment: PyiAssignment) -> bool:
    """Whether `assignment` is what `enum_alias_assignment` builds."""
    return (
        assignment.name is not None
        and assignment.statement == enum_alias_assignment(assignment.name).statement
    )


def convert_enum(node: Nodes.CEnumDefNode) -> PyiEnum | PyiAssignment:
    """Convert a Cython enum definition to PyiEnum."""
    if node.create_wrapper:  # type: ignore
        name: str | None = node.name  # type: ignore
        return PyiEnum(enum_name=name, names=get_enum_names(node))
    # Make it usable as an alias for int
    return enum_alias_assignment(node.name)  # type: ignore


def _is_unhashable_hash_assignment(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__hash__"
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    )


def _assignment_target_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    """The plain name a parsed `ast.Assign`/`ast.AnnAssign` targets, or
    `None` for anything else this doesn't cleanly apply to (a tuple/
    multiple-target assignment, an attribute target, etc.)."""
    target: ast.expr | None
    if isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        target = node.targets[0] if len(node.targets) == 1 else None
    return target.id if isinstance(target, ast.Name) else None
