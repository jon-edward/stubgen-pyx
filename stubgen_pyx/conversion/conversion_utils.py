"""Utility functions for converting Cython AST nodes."""

from __future__ import annotations

import logging
import textwrap


from Cython.Compiler import Nodes, ExprNodes


def extract_type_from_base_type(node) -> str | None:
    """Extract type name from a base_type node, trying multiple approaches."""
    base_type = node.base_type
    try:
        if isinstance(base_type, Nodes.CTupleBaseTypeNode):
            return _extract_tuple_base_type_node(base_type)
        name = None
        if hasattr(base_type, "name") and base_type.name is not None:
            name = ".".join(base_type.module_path + [base_type.name])
        if (
            hasattr(base_type, "base_type_node")
            and base_type.base_type_node is not None
        ):
            name = ".".join(
                base_type.base_type_node.module_path + [base_type.base_type_node.name]
            )
        if name == "char" and isinstance(node.declarator, Nodes.CPtrDeclaratorNode):
            # char * -> str, too complex to handle with name substitution
            return "str"
        return name
    except AttributeError:
        pass
    return None


# TODO: Handle templated types
# def _extract_templated_base_type_node(node) -> str | None:
#     print(node.dump())
#     base_type = extract_type_from_base_type(node.base_type_node)
#     print(base_type)
#     types = []
#     kwargs = {}
#     for arg in node.positional_args:
#         types.append(extract_type_from_base_type(arg.base_type))
#     if isinstance(node.keyword_args, ExprNodes.DictNode):
#         for key, value in node.keyword_args.key_value_pairs:
#             if not isinstance(key, ExprNodes.IdentifierStringNode):
#                 continue
#             kwargs[key.value] = unparse_expr(value)
#     parts = types + [f"{k}={v}" for k, v in kwargs.items()]
#     return f"{base_type}[{', '.join(parts)}]"


def _extract_tuple_base_type_node(node) -> str:
    output = []
    for base in node.components:
        output.append(extract_type_from_base_type(base) or "object")
    return f"tuple[{', '.join(output)}]"


def get_source(source: str, node: Nodes.Node) -> str:
    """Extract source code for a node, dedented and trimmed.

    Note: Node end_pos is often inaccurate; fallback to start position if needed.
    """
    lines = source.splitlines(keepends=True)

    end_pos = node.end_pos()
    if end_pos is None:
        end_pos = node.pos
    output = "".join(lines[i - 1] for i in range(node.pos[1], end_pos[1] + 1))
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
    if not hasattr(node, "bases") or not node.bases:  # type: ignore
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


def _get_name_from_ptr_or_name_decl(
    decl: Nodes.CNameDeclaratorNode
    | Nodes.CPtrDeclaratorNode
    | Nodes.CConstDeclaratorNode,
) -> str:
    if isinstance(
        decl,
        (
            Nodes.CPtrDeclaratorNode,
            Nodes.CConstDeclaratorNode,
            Nodes.CFuncDeclaratorNode,
        ),
    ):
        return _get_name_from_ptr_or_name_decl(decl.base)
    return decl.name


def get_cdef_variables(node: Nodes.CVarDefNode) -> list[tuple[str, str | None]]:
    """Extract all variable names from a cdef variable node.

    A single ``cdef public int x, y, z`` node has multiple declarators.
    Returns a list of (name, type) pairs, one per declarator.
    The type may be None if it cannot be determined.
    """
    base_type = extract_type_from_base_type(node)  # type: ignore
    if (
        not base_type
        and hasattr(node, "base_type")
        and isinstance(node.base_type, Nodes.CPtrDeclaratorNode)
    ):
        base_type = extract_type_from_base_type(node.base_type)
    accepted_declarators = (
        Nodes.CNameDeclaratorNode,
        Nodes.CPtrDeclaratorNode,
        Nodes.CConstDeclaratorNode,
        Nodes.CFuncDeclaratorNode,
    )
    declarators = []
    for declarator in node.declarators:
        if isinstance(declarator, accepted_declarators):
            declarators.append(declarator)
        else:
            logging.warning(f"Unknown declarator type: {type(declarator).__name__}")
    # TODO: Handle function pointers by properly extracting the return and argument types
    return [
        (
            _get_name_from_ptr_or_name_decl(d),
            base_type if not isinstance(d, Nodes.CFuncDeclaratorNode) else "Callable",
        )
        for d in declarators
    ]


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
