"""Extracts function signatures from Cython AST nodes."""

from __future__ import annotations

import logging

from Cython.Compiler import Nodes

from .conversion_utils import unparse_expr
from ..models.pyi_elements import PyiArgument, PyiSignature


logger = logging.getLogger(__name__)


def get_signature(node: Nodes.CFuncDefNode | Nodes.DefNode) -> PyiSignature:
    """Extract a PyiSignature from a Cython function node."""
    if isinstance(node, Nodes.CFuncDefNode):
        return _get_signature_cfunc(node)
    return _get_signature_def(node)


def _get_signature_def(node: Nodes.DefNode) -> PyiSignature:
    """Extract signature from a Python (def) function node."""
    pyi_args = _get_args(node.args)  # type: ignore

    var_arg = _create_argument_if_exists(node.star_arg)
    kw_arg = _create_argument_if_exists(node.starstar_arg)
    return_type = _get_return_type_annotation(node)

    return PyiSignature(
        pyi_args,
        var_arg=var_arg,
        kw_arg=kw_arg,
        return_type=return_type,
        num_posonly_args=node.num_posonly_args,
        num_kwonly_args=node.num_kwonly_args,
    )


def _get_signature_cfunc(node: Nodes.CFuncDefNode) -> PyiSignature:
    """Extract signature from a C (cdef/cpdef) function node."""
    pyi_args = _get_args(node.declarator.args)  # type: ignore
    return_type = _get_return_type_annotation(node)
    return PyiSignature(pyi_args, return_type=return_type)


def _create_argument_if_exists(arg_node) -> PyiArgument | None:
    """Convert an argument node to PyiArgument if it exists."""
    if arg_node is None:
        return None
    return PyiArgument(arg_node.name, annotation=_get_annotation(arg_node))


def _decode_or_pass(value: str | bytes) -> str:
    """Ensure value is a string, decoding bytes if needed."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected str or bytes, got {type(value)}")


def _extract_type_from_base_type(base_type) -> str | None:
    """Extract type name from a base_type node, trying multiple approaches."""
    try:
        if base_type.name is not None:
            return _decode_or_pass(base_type.name)
        if base_type.base_type_node is not None:
            name = ".".join(
                base_type.base_type_node.module_path + [base_type.base_type_node.name]
            )
            return name
    except AttributeError:
        pass
    return None


def _get_annotation(arg: Nodes.CArgDeclNode) -> str | None:
    """Extract type annotation from a function argument node."""
    try:
        if arg.annotation is not None:
            return _decode_or_pass(arg.annotation.string.value)
        return _extract_type_from_base_type(arg.base_type)
    except AttributeError:
        pass
    return None


def _get_return_type_annotation(node: Nodes.CFuncDefNode | Nodes.DefNode) -> str | None:
    """Extract return type annotation from a function node."""
    if node.return_type_annotation is not None:
        return _decode_or_pass(node.return_type_annotation.string.value)

    try:
        base_type = node.base_type  # type: ignore
        return _extract_type_from_base_type(base_type)
    except AttributeError:
        pass
    return None


def _to_argument(arg: Nodes.CArgDeclNode) -> PyiArgument:
    """Convert a CArgDeclNode to a PyiArgument."""
    declarator: Nodes.CDeclaratorNode | Nodes.CPtrDeclaratorNode = arg.declarator  # type: ignore
    if isinstance(declarator, Nodes.CPtrDeclaratorNode):
        name = _decode_or_pass(declarator.base.name)  # type: ignore
    else:
        name = _decode_or_pass(declarator.name)  # type: ignore
    if not name:
        name = arg.base_type.name  # type: ignore
        annotation = None
    else:
        annotation = _get_annotation(arg)

    default = unparse_expr(arg.default)  # type: ignore
    return PyiArgument(name, default=default, annotation=annotation)


def _get_args(args: list[Nodes.CArgDeclNode]) -> list[PyiArgument]:
    """Convert a list of CArgDeclNodes to PyiArguments."""
    return [_to_argument(arg) for arg in args]
