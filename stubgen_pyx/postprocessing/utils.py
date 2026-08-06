"""
Shared postprocessing utilities.
"""

import ast


def dotted_name(node: ast.expr) -> str:
    """Return the dotted name of an expression, or ``""`` if it has none."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""
