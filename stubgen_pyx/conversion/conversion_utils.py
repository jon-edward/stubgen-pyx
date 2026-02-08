"""Utility functions for converting Cython AST nodes."""

from __future__ import annotations

import textwrap

from Cython.Compiler import Nodes, ExprNodes


class _LinesCache:
    """Cache for source code lines to avoid repeated splits."""

    def __init__(self):
        self._source: str | None = None
        self.lines: list[str] = []

    @property
    def source(self) -> str | None:
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        if value == self._source:
            return
        self._source = value
        self.lines = value.splitlines(keepends=True)


_lines_cache = _LinesCache()


def get_source(source: str, node: Nodes.Node) -> str:
    """Extract source code for a node, dedented and trimmed.

    Note: Node end_pos is often inaccurate; fallback to start position if needed.
    """
    _lines_cache.source = source
    lines = _lines_cache.lines

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
    """Extract decorator expressions from a function or class node."""
    if node.decorators:
        return [get_source(source, node) for node in node.decorators]
    return []


def get_bases(node: Nodes.CClassDefNode | Nodes.PyClassDefNode) -> list[str]:
    """Extract base class names from a class node."""
    if not node.bases:  # type: ignore
        return []
    output = []
    for base in node.bases.args:  # type: ignore
        if isinstance(base, ExprNodes.NameNode):
            output.append(base.name)  # type: ignore
    return output


def get_metaclass(node: Nodes.PyClassDefNode | Nodes.CClassDefNode) -> str | None:
    """Extract metaclass name from a Python class node, if present."""
    if not isinstance(node, Nodes.PyClassDefNode):
        return None
    if node.metaclass and isinstance(node.metaclass, ExprNodes.NameNode):
        return node.metaclass.name  # type: ignore
    return None


def get_enum_names(node: Nodes.CEnumDefNode) -> list[str]:
    """Extract member names from an enum definition node."""
    return [item.name for item in node.items]  # type: ignore


def docstring_to_string(docstring: str) -> str:
    """Convert a raw docstring to a Python string literal with triple quotes."""
    if not docstring:
        return '""" """'
    first_line, *rest = docstring.splitlines(keepends=True)
    rest_joined = textwrap.dedent("".join(rest))
    docstring = f"{first_line}{rest_joined}".replace('"""', r"\"\"\"")
    return f'"""{docstring}"""'


def unparse_expr(node: Nodes.Node | None) -> str | None:
    """Convert a default argument expression to source code string.

    Simple literals are unparsed to their string form. Complex expressions
    are replaced with '...'.
    """
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
