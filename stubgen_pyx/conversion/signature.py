"""
Converts Cython signature nodes to PyiSignature.
"""

from __future__ import annotations

import logging

from Cython.Compiler import Nodes

from .conversion_utils import unparse_expr
from ..models.pyi_elements import PyiArgument, PyiSignature


logger = logging.getLogger(__name__)


def get_signature(node: Nodes.CFuncDefNode | Nodes.DefNode) -> PyiSignature:
    """Gets the signature for a Cython AST node."""
    if isinstance(node, Nodes.CFuncDefNode):
        return _get_signature_cfunc(node)
    return _get_signature_def(node)


def _get_signature_def(node: Nodes.DefNode) -> PyiSignature:
    """Gets the signature for a DefNode."""
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
    """Gets the signature for a CFuncDefNode."""
    pyi_args = _get_args(node.declarator.args)  # type: ignore
    return_type = _get_return_type_annotation(node)
    return PyiSignature(pyi_args, return_type=return_type)


def _create_argument_if_exists(arg_node) -> PyiArgument | None:
    """Creates an Argument from a node if it exists, otherwise returns None."""
    if arg_node is None:
        return None
    return PyiArgument(arg_node.name, annotation=_get_annotation(arg_node))


def _decode_or_pass(value: str | bytes) -> str:
    """Decodes bytes to string, or returns string as-is."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _extract_type_from_base_type(base_type) -> str | None:
    """Extracts type name from a base_type node."""
    try:
        if base_type.name is not None:
            return _decode_or_pass(base_type.name)
        if base_type.base_type_node is not None:
            name = f".".join(
                base_type.base_type_node.module_path + [base_type.base_type_node.name]
            )
            return name
    except AttributeError:
        pass
    return None


def _get_annotation(arg: Nodes.CArgDeclNode) -> str | None:
    """Extracts annotation from a CArgDeclNode."""
    try:
        if arg.annotation is not None:
            return _decode_or_pass(arg.annotation.string.value)
        return _extract_type_from_base_type(arg.base_type)
    except AttributeError:
        pass
    return None


def _get_return_type_annotation(node: Nodes.CFuncDefNode | Nodes.DefNode) -> str | None:
    """Extracts return type annotation from a function node."""
    if node.return_type_annotation is not None:
        return _decode_or_pass(node.return_type_annotation.string.value)

    try:
        base_type = node.base_type  # type: ignore
        return _extract_type_from_base_type(base_type)
    except AttributeError:
        pass
    return None


def _to_argument(arg: Nodes.CArgDeclNode) -> PyiArgument:
    """Converts a CArgDeclNode to an Argument."""
    name: str = _decode_or_pass(arg.declarator.name)  # type: ignore
    if not name:
        name = arg.base_type.name  # type: ignore
        annotation = None
    else:
        annotation = _get_annotation(arg)

    default = unparse_expr(arg.default)  # type: ignore
    return PyiArgument(name, default=default, annotation=annotation)


def _get_args(args: list[Nodes.CArgDeclNode]) -> list[PyiArgument]:
    """Converts a list of CArgDeclNodes to a tuple of Arguments."""
    return [_to_argument(arg) for arg in args]
