"""C declarator helpers for conversion."""

from __future__ import annotations

import logging

from Cython.Compiler import Nodes

from .type_parsing import extract_type_from_base_type

_logger = logging.getLogger(__name__)


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
        ),
    ):
        return _declarator_name(decl.base)
    return getattr(decl, "name", None)


def _get_func_decl_type(
    decl: Nodes.CFuncDeclaratorNode, base_type: str | None
) -> str | None:
    if not isinstance(decl, Nodes.CFuncDeclaratorNode):
        return None

    args = [extract_type_from_base_type(a) or "Incomplete" for a in decl.args]
    return f"Callable[[{', '.join(args)}], {base_type or 'Incomplete '}]"


def get_cdef_declarator_type(
    decl, base_type: str | None = None
) -> tuple[str | None, str | None]:
    name = _declarator_name(decl)
    if isinstance(decl, Nodes.CFuncDeclaratorNode):
        typ = _get_func_decl_type(decl, base_type) or "Callable[..., Any]"
    elif isinstance(decl, Nodes.CPtrDeclaratorNode):
        decl = decl.base
        _, typ = get_cdef_declarator_type(decl, base_type)
    else:
        typ = extract_type_from_base_type(decl)
    return (name, typ or base_type)


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
    )
    declarators = []
    for decl in node.declarators:
        if isinstance(decl, accepted):
            declarators.append(decl)
        else:
            _logger.warning("Unknown declarator type: %s", type(decl).__name__)

    is_ptr = (
        False
        if not declarators
        else isinstance(declarators[0], Nodes.CPtrDeclaratorNode)
    )
    base_type = extract_type_from_base_type(node, is_ptr=is_ptr)

    results = []
    for d in declarators:
        results.append(get_cdef_declarator_type(d, base_type=base_type))
    return results


def get_enum_names(node: Nodes.CEnumDefNode) -> list[str]:
    """Return member names from an enum definition node."""
    return [item.name for item in node.items]  # type: ignore
