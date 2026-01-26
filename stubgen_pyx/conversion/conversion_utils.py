"""
Provides utility functions for converting Cython AST nodes to PyiElements.
"""

from __future__ import annotations

import textwrap

from Cython.Compiler import Nodes, ExprNodes


def get_source(source: str, node: Nodes.Node) -> str:
    """Gets the source code lines for a Cython AST node.

    The calculated end_pos is often inaccurate.
    """
    lines = source.splitlines(keepends=True)
    end_pos = node.end_pos()
    if end_pos is None:
        end_pos = node.pos
    output = ""
    for i in range(node.pos[1], end_pos[1] + 1):
        output += lines[i - 1]
    return textwrap.dedent(output).rstrip()


def get_decorators(
    source: str,
    node: Nodes.DefNode
    | Nodes.CFuncDefNode
    | Nodes.CClassDefNode
    | Nodes.PyClassDefNode,
) -> list[str]:
    """Gets the decorators for a Cython AST node."""
    if node.decorators:
        return [get_source(source, node) for node in node.decorators]
    return []


def get_bases(node: Nodes.CClassDefNode | Nodes.PyClassDefNode) -> list[str]:
    """Gets the bases for a Cython AST node."""
    if not node.bases:  # type: ignore
        return []
    output = []
    for base in node.bases.args:  # type: ignore
        if isinstance(base, ExprNodes.NameNode):
            output.append(base.name)  # type: ignore
    return output


def get_metaclass(node: Nodes.PyClassDefNode | Nodes.CClassDefNode) -> str | None:
    """Gets the metaclass for a Cython AST node."""
    if not isinstance(node, Nodes.PyClassDefNode):
        return None
    if node.metaclass and isinstance(node.metaclass, ExprNodes.NameNode):
        return node.metaclass.name  # type: ignore
    return None


def get_enum_names(node: Nodes.CEnumDefNode) -> list[str]:
    """Gets the enum names for a Cython AST node."""
    return [item.name for item in node.items]  # type: ignore


def docstring_to_string(docstring: str) -> str:
    """Converts a Cython docstring to a Python docstring."""
    first_line, *rest = docstring.splitlines(keepends=True)
    rest_joined = textwrap.dedent("".join(rest))
    docstring = f"{first_line}{rest_joined}".replace('"""', r"\"\"\"")
    return f'"""{docstring}"""'


def unparse_expr(node: Nodes.Node | None) -> str | None:
    """Unparse a default argument. Returns '...' for complex expressions that can't be unparsed."""
    if node is None:
        return None

    if isinstance(node, ExprNodes.NoneNode):
        return "None"
    if isinstance(node, ExprNodes.NameNode):
        return node.name  # type: ignore
    if isinstance(node, (ExprNodes.IntNode, ExprNodes.FloatNode, ExprNodes.BoolNode)):
        return node.value  # type: ignore
    if isinstance(node, (ExprNodes.UnicodeNode, ExprNodes.BytesNode)):
        return repr(node.value)  # type: ignore
    return "..."
