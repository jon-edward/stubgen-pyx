"""C declarator helpers for conversion."""

from __future__ import annotations

import logging
from Cython.Compiler import Nodes

from .type_parsing import extract_type_from_base_type


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
